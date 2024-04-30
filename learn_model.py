import torch
import numpy as np
import random
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from dataset import create_dataset
from new_model import GNN, GNN_noMMP
from plots import trajectorytogif
from torch.profiler import profile, record_function, ProfilerActivity
from coarse_interpolate_utils import create_maps_distances
import time


stopwatch = 1



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
            # self.net.load_state_dict(torch.load('checkpoints/chk_83209', map_location = self.device))
            self.net.load_state_dict(torch.load('checkpoints/chk_AE_83209', map_location = self.device))

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

                if self.multi_scaling:
                    # create just once maps in order to improve performance
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list = create_maps_distances(node_positions, self.scales, edge_index)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list)

                
                # Profiling outside the inner loop to profile across one batch per epoch for performance overview
                # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                # Iterate over each timestep in the trajectory
                for batch in range(0, trajs_size - 1, self.batch_size):
                    u_batch = u[batch:batch + self.batch_size]
                    # u_batch[:, in_nodes, 0] += (self.noise_var) ** (0.5) * torch.randn_like(u_batch[:, in_nodes, 0])

                    torch.set_printoptions(precision=10, threshold=5000, edgeitems=10, linewidth=200)

                    # Forward pass: Pass the current timestep to the GNN
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
                    print(f"Simulation {sim}: partial time: {partial-start_time}")

    
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
        torch.save(self.net.state_dict(), f'checkpoints/chk_AE_{idf}')
        print("Saving model")


    # def train(self):
    #     ''' Trains the model'''
    #     rollout_train_loss = []
    #     trajs_size = self.train_data['trajs'][0].shape[0]
    #     print("Start Training")
    #     for epoch in range(self.epochs):
    #         rollout_train_loss.clear()
    #         #shuffle data
    #         indices = list(range(len(self.train_data['trajs'])))
    #         random.shuffle(indices)
    #         # training
    #         for sim in indices:
    #             u = self.train_data['trajs'][sim]
    #             edge_index = self.train_data['edge_index'][sim]
    #             edge_weights = self.train_data['edge_weights'][sim].repeat(self.batch_size - 1, 1, 1)
    #             in_nodes = self.train_data['in_nodes'][sim]

    #             for batch in range(0, trajs_size - 1, self.batch_size):
    #                 u_batch = u[batch:batch + self.batch_size]
    #                 target = u_batch
    #                 # add gaussian noise
    #                 u_batch[:, in_nodes, 0] += (self.noise_var) ** (0.5) * torch.randn_like(u_batch[:, in_nodes, 0])

    #                 # forward pass
    #                 du_net = self.net(u_batch[:self.batch_size - 1], edge_index, edge_weights)  # (bs,nodes,1)
    #                 du = (target[1:, :, 0] - u_batch[:-1, :, 0]) / self.dt
    #                 train_loss_1 = ((du_net[:, :, 0] - du) ** 2)[:, in_nodes].mean()
    #                 u_net = u_batch[:-1, :, 0] + self.dt * du_net[:, :, 0]
    #                 if self.problem == 'Stokes':
    #                     u_net = u_net*(u_net>0) # The solution of Stokes problem
    #                                             # must be always non negative
    #                 train_loss_2 = ((u_net - target[1:, :, 0]) ** 2)[:, in_nodes].mean()
    #                 train_loss = self.w1*train_loss_1 + self.w2*train_loss_2
    #                 rollout_train_loss.append(train_loss.item())
    #                 # backpropagation
    #                 self.optimizer.zero_grad()
    #                 train_loss.backward()
    #                 self.optimizer.step()

    #         self.scheduler.step()

    #         # print rollout number and MSE for training set at each epoch
    #         mse_train = sum(rollout_train_loss) / len(rollout_train_loss)
    #         print(f"Epoch {epoch+1:1f}: MSE_train {mse_train :6.6f}")

    #     print("End Training")
    #     print("Saving model")
    #     idf = np.random.randint(100000)
    #     torch.save(self.net.state_dict(), f'checkpoints/chk_{idf}')

    # def forecast(self, save_plot=True):
    #     print("Start Evaluation")
    #     for sim in range(len(self.test_data['trajs'])):
    #         mesh = self.test_data['mesh'][sim]

    #         node_positions = self.test_data['in_nodes'][sim]
    #         node_positions = torch.tensor(node_positions, dtype=torch.float)

    #         u = self.test_data['trajs'][sim]
    #         edge_index = self.test_data['edge_index'][sim]
    #         edge_attr = self.test_data['edge_weights'][sim].unsqueeze(0)
    #         total_error = 0

    #         # Iterate over each timestep
    #         for t in range(u.shape[0]):
    #             u_t = u[t:t+1]  # Current timestep
    #             # print("sk")
    #             # print(u_t.shape)
                
    #             # Network output for the current timestep
    #             # output_t = self.net(u_t, edge_index, edge_attr)
    #             output_t = self.net(u_t, edge_index, edge_attr, node_positions)
    #             # print("sk1")
    #             # # print(output_t)
    #             # print(output_t.shape)

    #             # Calculate reconstruction error for the current timestep
    #             error_t = ((output_t[:,:,0] - u_t[:,:,0]) ** 2).mean().item()
    #             total_error += error_t

    #         avg_error = total_error / u.shape[0]
    #         print(f"Test simulation {sim+1}: Average Reconstruction Error {avg_error:.6f}")

    #         # if save_plot:
    #         #     # Optional: Implement visualization of input vs. reconstructed output for each simulation
    #         #     print(f"Visualizing input and output for simulation {sim+1}")

    #         if save_plot:
    #             # Assuming trajectorytogif function is adapted to handle single timestep visualizations
    #             trajectorytogif(output_t.cpu().detach(), self.dt, name=f"images/test_sim_{sim}_output_{self.problem}", mesh=mesh)
    #             trajectorytogif(u_t.cpu().detach(), self.dt, name=f"images/test_sim_{sim}_input_{self.problem}", mesh=mesh)


    #     print("End Evaluation") 

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

            # Maps creation for performance, assuming similar need as in training
            # fine2coarse_list, distances_list = create_maps_distances(node_positions, self.scales)
            # self.net.assign_maps_coarsening(fine2coarse_list, distances_list)
            fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list = create_maps_distances(node_positions, self.scales, edge_index)
            self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list)

            output_t = torch.zeros(u.shape).to(self.device)

            # Iterate over each timestep
            for t in range(u.shape[0]):
                u_t = u[t:t+1]  # Current timestep

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


    # def forecast(self, save_plot = True):
    #     ''' Performs simulation rollout across all test simulations '''
    #     steps = self.test_data['trajs'][0].shape[0] - 1
    #     print("Start Testing")
    #     for sim in range(len(self.test_data['trajs'])):
    #         u = self.test_data['trajs'][sim]
    #         edge_index = self.test_data['edge_index'][sim]
    #         edge_attr = self.test_data['edge_weights'][sim].unsqueeze(0)
    #         b_nodes = self.test_data['b_nodes'][sim]
    #         mesh = self.test_data['mesh'][sim]
    #         u_net = torch.zeros(u.shape).to(self.device)
    #         u_net[0] = u[0]
    #         u0 = u_net[[0]]

    #         for i in range(steps):
    #             du_net = self.net(u0, edge_index, edge_attr)
    #             u1 = u0 + self.dt*du_net
    #             if self.problem == 'Stokes':
    #                 u1 = u1*(u1>0)
    #             u1[0,b_nodes,0] = u[i+1,b_nodes,0]
    #             u1[0,:,1:] = u[i+1,:,1:]
    #             u_net[i+1] = u1[0].detach()
    #             u0 = u1.detach()

    #         error = (((u_net[:,:,0]-u[:,:,0])**2).sum(1)/(u[:,:,0]**2).sum(1)).mean()
    #         print(f"Test simulation {sim+1:1f}: RMSE {error :6.6f}")
    #         if save_plot:
    #             trajectorytogif(u_net, self.dt, name=f"images/test_sim_{sim}_pred_"+f"{self.problem}", mesh=mesh)
    #             trajectorytogif(u, self.dt, name=f"images/test_sim_{sim}_true_"+f"{self.problem}", mesh=mesh)


    #     print("End Testing")
