import torch
import numpy as np
import random
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from dataset import create_dataset, random_edge_augmentation

from plots import trajectorytogif
from torch.profiler import profile, record_function, ProfilerActivity
from geom_coarse_interpolate_utils import create_maps_distances
import time


stopwatch = 1

old_model = 1

if old_model:
    from geom_new_model import GNN
else:
    from geom_model import GNN, GNN_noMMP



class Learner():
    ''' Class used for model training and rollout prediction'''
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Problem
        self.problem = args.example

        self.scales = args.scales

        self.multi_scaling = args.multi_scaling

        if self.multi_scaling:
            self.net = GNN(args).to(self.device)
        else:
            self.net = GNN_noMMP(args).to(self.device)
        
        if not args.train_model:
            # self.net.load_state_dict(torch.load('checkpoints/pretrained_net_' + f'{self.problem}', map_location = self.device))
            # self.net.load_state_dict(torch.load('checkpoints/chk_AE_83209', map_location = self.device))
            self.net.load_state_dict(torch.load('checkpoints/chk_AE_79691', map_location = self.device))

        if args.resume_training:
            self.net.load_state_dict(torch.load('checkpoints/chk_AE_83209', map_location = self.device))

        # Training parameters
        if self.problem == 'AD':
            print("Using default dt for AD")
            self.dt = 0.02
        if self.problem == 'Stokes':
            print("Using default dt for Stokes")
            self.dt = 0.01
        self.lr = args.lr
        self.milestones = args.milestones
        self.noise_var = args.noise_var
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.w1 = args.w1
        self.w2 = args.w2
        self.optimizer = Adam(self.net.parameters(), self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        self.train_size = args.train_size

        self.augment_factor = args.edge_augment_factor

        # Dataset creation
        self.train_data, self.test_data = create_dataset(self.device, self.problem, self.train_size)

    def train(self):
        print("Start Training")
        trajs_size = self.train_data['trajs'][0].shape[0]
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0
            # Assuming self.train_data is accessible and properly formatted
            for sim in range(len(self.train_data['trajs'])):

                u = self.train_data['trajs'][sim]  # Full trajectory for the current simulation
                # node_positions = torch.tensor(self.train_data['in_nodes'][sim], dtype=torch.float)
                if not isinstance(self.train_data['in_nodes'][sim], np.ndarray):
                    node_positions_array = np.stack(self.train_data['in_nodes'][sim], axis=0)
                else:
                    node_positions_array = self.train_data['in_nodes'][sim]
                node_positions = torch.tensor(node_positions_array, dtype=torch.float).to(self.device)
                edge_index = self.train_data['edge_index'][sim]
                # edge_attr = self.train_data['edge_weights'][sim].unsqueeze(0)
                edge_attr = self.train_data['edge_weights'][sim].repeat(self.batch_size, 1, 1)

                mesh = self.train_data['mesh'][sim]

                if self.augment_factor > 0:
                    edge_index, edge_attr = random_edge_augmentation(edge_index, edge_attr, mesh, augment_factor=self.augment_factor)

                if old_model:
                    if self.multi_scaling:
                        # create just once maps in order to improve performance
                        fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index)
                        self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)

                
                # Profiling outside the inner loop to profile across one batch per epoch for performance overview
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                # Iterate over each timestep in the trajectory
                for batch in range(0, trajs_size - 1, self.batch_size):
                    u_batch = u[batch:batch + self.batch_size]
                    u_batch += (self.noise_var) ** (0.5) * torch.randn_like(u_batch)
                    # u_batch[:, in_nodes, 0] += (self.noise_var) ** (0.5) * torch.randn_like(u_batch[:, in_nodes, 0])

                    torch.set_printoptions(precision=10, threshold=5000, edgeitems=10, linewidth=200)

                    # Forward pass: Pass the current timestep to the GNN
                    if old_model:
                        output = (self.net(u_batch.squeeze(0), edge_index, edge_attr.squeeze(0), node_positions)).unsqueeze(0)
                    else:
                        output = self.net(u_batch, edge_index, edge_attr, node_positions)
                    
                    
                    # Calculate loss as the difference between GNN output and the next timestep
                    # print('u_batch: ', type(u_batch), u_batch.size(), '\n')
                    # print('output: ', type(output), output.size(), '\n')
                    loss = ((output[:,:,0] - u_batch[:,:,0]) ** 2).mean()
                    
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()

                if stopwatch:
                    partial = time.time()
                    print(f"Simulation {sim}: partial time: {partial-start_time:.6f} Loss = {loss.item():.6f}")

    
            # Scheduler step (if using LR scheduler)
            self.scheduler.step()

            # Print average loss for the epoch
            avg_loss = total_loss / (len(self.train_data['trajs']) * (u.shape[0] - 1))
            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")

        print("End Training")
        # Save the model
        idf = np.random.randint(100000)
        torch.save(self.net.state_dict(), f'/content/GNN_autoencoder/checkpoints/chk_AE_{idf}')
        print("Saving model")


    

    def forecast(self, save_plot=True):
        print("Start Evaluation")
        for sim in range(len(self.test_data['trajs'])):
            mesh = self.test_data['mesh'][sim]

            # Handling node positions with similar conditional check as in train method
            if not isinstance(self.test_data['in_nodes'][sim], np.ndarray):
                node_positions_array = np.stack(self.test_data['in_nodes'][sim], axis=0)
            else:
                node_positions_array = self.test_data['in_nodes'][sim]
            node_positions = torch.tensor(node_positions_array, dtype=torch.float).to(self.device)

            u = self.test_data['trajs'][sim]
            edge_index = self.test_data['edge_index'][sim]
            # Adjusting edge attributes handling to match the train method
            # edge_attr = self.test_data['edge_weights'][sim].repeat(self.batch_size, 1, 1)
            edge_attr = self.test_data['edge_weights'][sim].unsqueeze(0)

            total_error = 0

            if old_model:
                if self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)
                

            mesh = self.test_data['mesh'][sim]

            if self.augment_factor > 0:
                edge_index, edge_attr = random_edge_augmentation(edge_index, edge_attr, mesh, augment_factor=self.augment_factor)

            output_t = torch.zeros(u.shape).to(self.device)

            # Iterate over each timestep
            for t in range(u.shape[0]):
                u_t = u[t:t+1]  # Current timestep

                if old_model:
                    output_t[t,:,:] = (self.net(u_t.squeeze(0), edge_index, edge_attr.squeeze(0), node_positions)).unsqueeze(0)

                else:
                    # Forward pass with the adjusted inputs
                    output_t[t, :,:] = self.net(u_t, edge_index, edge_attr, node_positions)

                # Calculate reconstruction error for the current timestep
                error_t = ((output_t[t,:,0] - u_t[:,:,0]) ** 2).mean().item()
                total_error += error_t

            avg_error = total_error / u.shape[0]
            print(f"Test simulation {sim+1}: Average Reconstruction Error {avg_error:.6f}")

            if save_plot:
                # Visualizing and saving outputs, ensure method is adapted for batch or single timestep visualizations
                trajectorytogif(output_t.cpu().detach(), self.dt, name=f"images/test_sim_{sim}_output_{self.problem}", mesh=mesh)
                trajectorytogif(u.cpu().detach(), self.dt, name=f"images/test_sim_{sim}_input_{self.problem}", mesh=mesh)

        print("End Evaluation")





   
