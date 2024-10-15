import torch
import numpy as np
import random
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from dataset import create_dataset, random_edge_augmentation

from plot_utils.plots import trajectorytogif
from torch.profiler import profile, record_function, ProfilerActivity
from geom_coarse_interpolate_utils import create_maps_distances
import time
import copy
from PIL import Image
from plot_utils.plot_pooled import plot_pooled

import math


from torch_geometric.loader import DataLoader

from torch.cuda import amp



stopwatch = 1

new_model = 1

if new_model:
    from geom_new_model import GNN
else:
    from old_scripts.geom_model import GNN, GNN_noMMP



class Learner():
    ''' Class used for model training and rollout prediction'''
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Problem
        self.problem = args.example

        self.scales = args.scales
        self.checkpoint = args.checkpoint
        self.checkpoint_ROM = args.checkpoint_ROM
        self.multi_scaling = args.multi_scaling

        self.resume_training = args.resume_training

        if self.multi_scaling:
            self.net = GNN(args).to(self.device)
            self.net_ROM = GNN(args).to(self.device)
        else:
            self.net = GNN_noMMP(args).to(self.device)
        
        if not args.train_model:
            # self.net.load_state_dict(torch.load('checkpoints/pretrained_net_' + f'{self.problem}', map_location = self.device))
            # self.net.load_state_dict(torch.load('checkpoints/chk_AE_83209', map_location = self.device))
            self.net.load_state_dict(torch.load(self.checkpoint, map_location = self.device))

        if args.test_ROM:
            self.net_ROM.load_state_dict(torch.load(self.checkpoint_ROM, map_location = self.device))

        if args.resume_training:
            self.net.load_state_dict(torch.load(self.checkpoint, map_location = self.device))

        # Training parameters
        if self.problem == 'AD':
            print("Using default dt for AD")
            self.dt = 0.02
        if self.problem == 'Stokes':
            print("Using default dt for Stokes")
            self.dt = 0.01
        if self.problem == 'NS':
            print("Using default dt for NS")
            T = 3.5
            num_steps = int(1000*T)
            dt = T/num_steps
            self.dt = dt
        self.lr = args.lr
        self.milestones = args.milestones
        self.noise_var = args.noise_var
        self.ROM = args.ROM
        self.w1 = args.w1
        self.w2 = args.w2
        if self.ROM:
            self.noise_var = 0
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.w1 = args.w1
        self.w2 = args.w2
        self.pool_k = args.pool_k
        if args.train_model:
            self.optimizer = Adam(self.net.parameters(), self.lr)
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        if args.train_ROM:
            self.optimizer = Adam(self.net_ROM.parameters(), self.lr)
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)
        self.train_size = args.train_size

        self.augment_factor = args.edge_augment_factor

        # Dataset creation
        self.train_data, self.test_data = create_dataset(self.device, self.problem, self.train_size, args.train_model)

############################################################################################################
############################################################################################################

    def train(self):
        print("Start Training")

        idf = np.random.randint(100000)
        
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0

            for i, traj_data in enumerate(self.train_data):  # Iterate over trajectories
                traj_data = traj_data[:-1]  # Remove the last timestep from the trajectory

                traj_loader = DataLoader(traj_data, batch_size=self.batch_size)  # Create a DataLoader for the current trajectory

                node_positions= traj_data[0].pos

                # Create an iterator from traj_loader
                traj_loader_iter1 = iter(traj_loader)

                # Get the first batch of data
                first_batch = next(traj_loader_iter1)

                # Get edge_index from the first batch
                edge_index = first_batch.edge_index

                if new_model and self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index, self.batch_size)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)


                # Create an iterator from traj_loader
                traj_loader_iter = iter(traj_loader)

                trajectory_loss = 0

                time_batch = 0

                # Iterate over mini-batches of time steps
                while True:
                    try:
                        data = copy.deepcopy(next(traj_loader_iter))
                    except StopIteration:
                        break  # Exit the loop if there are no more batches

                    # if self.problem == 'NS':
                    #     x_even = data.x[0::2]
                    #     x_odd = data.x[1::2]
                    #     data.x =torch.hstack((x_even, x_odd))


                    data.x += (self.noise_var) ** (0.5) * torch.randn_like(data.x)
                    u = data.x

                    if self.ROM:
                        x_size = int(data.x.size(0) / self.batch_size)  # Calculate x_size, which is 774.0 in your case
                        time_vector = self.dt * torch.arange(self.batch_size).float().to(self.device) + time_batch  # Create time_vector
                        time_batch += self.dt * self.batch_size  # Update time_batch for the next iteration
                        rom_x = time_vector.repeat_interleave(x_size).unsqueeze(1)
                        data.x = rom_x

                       


                    # Forward pass: Pass the current timestep to the GNN
                    if new_model:
                        output = self.net(data)
                    else:
                        output = self.net(data.x, data.edge_index, data.edge_attr, data.pos)

                    # Calculate loss as the difference between GNN output and the next timestep
                    loss = ((output[:,0] - u[:,0])**2).mean()

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    trajectory_loss += loss.item()

                    total_loss += loss.item()

                trajectory_loss /= self.batch_size

                if stopwatch:
                    partial = time.time()
                    print(f"Simulation {i}: partial time: {partial-start_time:.6f} Trajectory Loss = {trajectory_loss:.6f}")

            # Scheduler step (if using LR scheduler)
            self.scheduler.step()

            # Print average loss for the epoch
            avg_loss = total_loss / (len(self.train_data)* self.batch_size)
            
            with open('loss_values.txt', 'a') as loss_file:
                loss_file.write(f"{avg_loss:.6f}\n")  # Write the loss value to the file

            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")

            if epoch % 10 == 0:
                torch.save(self.net.state_dict(), f'checkpoints/chk_ENC_DEC_rom{self.ROM}_{idf}_{epoch}_pool{self.pool_k}_sim_resumed_{self.resume_training}_batch_size_{self.batch_size}')
                print("Saving model")

        print("End Training")
        # Save the model
        idf = np.random.randint(100000)
        torch.save(self.net.state_dict(), f'checkpoints/chk_AE_{idf}')
        print("Saving model")

    def train_ROM(self):
        print("Start Training ROM")

        idf = np.random.randint(100000)
        
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0

            for i, traj_data in enumerate(self.train_data):

                traj_data = traj_data[:-1]  # Remove the last timestep from the trajectory

                traj_loader = DataLoader(traj_data, batch_size=self.batch_size)  # Create a DataLoader for the current trajectory

                node_positions= traj_data[0].pos

                # Create an iterator from traj_loader
                traj_loader_iter1 = iter(traj_loader)

                # Get the first batch of data
                first_batch = next(traj_loader_iter1)

                # Get edge_index from the first batch
                edge_index = first_batch.edge_index

                if new_model and self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index, self.batch_size)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)
                    self.net_ROM.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)


                # Create an iterator from traj_loader
                traj_loader_iter = iter(traj_loader)

                trajectory_loss = 0

                time_batch = 0

                # Iterate over mini-batches of time steps
                while True:
                    try:
                        data = copy.deepcopy(next(traj_loader_iter))
                    except StopIteration:
                        break  # Exit the loop if there are no more batches

                    # if self.problem == 'NS':
                    #     x_even = data.x[0::2]
                    #     x_odd = data.x[1::2]
                    #     data.x =torch.hstack((x_even, x_odd))


                    data.x += (self.noise_var) ** (0.5) * torch.randn_like(data.x)
                    u = data.x

                    
                    x_size = int(data.x.size(0) / self.batch_size)  # Calculate x_size, which is 774.0 in your case
                    time_vector = self.dt * torch.arange(self.batch_size).float().to(self.device) + time_batch  # Create time_vector
                    time_batch += self.dt * self.batch_size  # Update time_batch for the next iteration
                    rom_x = time_vector.repeat_interleave(x_size).unsqueeze(1)
                    data.x = rom_x

                       
                    # Deep copy of data since I need it twice
                    data_AE = copy.deepcopy(data)

                    # Encoder pass: Pass the current timestep to the GNN
                    data_AE, pool_indices_AE, pool_edges_AE, _, n_nodes_AE, _ = self.net.encoder(data_AE)

                    data_ROM, pool_indices_ROM, pool_edges_ROM, _, n_nodes_ROM, _ = self.net_ROM.encoder(data)

                    n_nodes = data_AE.x.size(0)
                    n_edges = data_AE.edge_index.size(0)*data_AE.edge_index.size(1)
                    # print(f'Number of nodes: {n_nodes}, Number of edges: {n_edges}')

                    # Call equalize_graph() function to equilize the graphs

                    data_AE, data_ROM = self.equalize_graph(data_AE, data_ROM, pool_indices_AE, pool_indices_ROM, pool_edges_AE, pool_edges_ROM)
                    

                    # Calculate loss as the difference between GNN output and the next timestep
                    loss = ((data_AE.x - data_ROM.x)**2).sum()/n_nodes + ((data_AE.edge_attr-data_ROM.edge_attr)**2).sum()/n_edges

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    trajectory_loss += loss.item()

                    total_loss += loss.item()

                trajectory_loss /= self.batch_size

                if stopwatch:
                    partial = time.time()
                    print(f"Simulation {i}: partial time: {partial-start_time:.6f} Trajectory Loss = {trajectory_loss:.6f}")

            # Scheduler step (if using LR scheduler)
            self.scheduler.step()

            # Print average loss for the epoch
            avg_loss = total_loss / (len(self.train_data)* self.batch_size)
            
            with open('loss_values.txt', 'a') as loss_file:
                loss_file.write(f"{avg_loss:.6f}\n")  # Write the loss value to the file

            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")

            if epoch % 10 == 0:
                torch.save(self.net.state_dict(), f'checkpoints/chk_train_ROM_{idf}_{epoch}')
                print("Saving model")

        print("End Training")
        # Save the model
        idf = np.random.randint(100000)
        torch.save(self.net.state_dict(), f'checkpoints/chk_AE_{idf}')
        print("Saving model")


    def equalize_graph(self, data_AE, data_ROM, pool_indices_AE, pool_indices_ROM, pool_edges_AE, pool_edges_ROM):
        
        # con        # confront pool_indices_AE and pool_indices_ROM
        # if an index is present in pool_indices_AE but not in pool_indices_ROM, add it to pool_indices_ROM and vice versa
        # length is equal
        
        # Find the indices where pool_indices_AE and pool_indices_ROM differ
        diff_indices = (pool_indices_AE != pool_indices_ROM).nonzero(as_tuple=True)[0]
        
        # Create masks for insertion
        mask_AE = pool_indices_AE[diff_indices] < pool_indices_ROM[diff_indices]
        mask_ROM = pool_indices_AE[diff_indices] > pool_indices_ROM[diff_indices]
        
        # Insert elements into pool_indices_ROM and data_ROM.x
        if mask_AE.any():
            insert_indices_AE = diff_indices[mask_AE]
            for idx in insert_indices_AE:
                pool_indices_ROM = torch.cat((pool_indices_ROM[:idx], pool_indices_AE[idx].unsqueeze(0), pool_indices_ROM[idx:]))
                data_ROM.x = torch.cat((data_ROM.x[:idx], torch.zeros_like(data_ROM.x[0]).unsqueeze(0), data_ROM.x[idx:]), 0)
        
        # if mask_AE.any():
        #     insert_indices_AE = diff_indices[mask_AE]
        #     pool_indices_ROM = torch.cat((pool_indices_ROM[:insert_indices_AE[0]], pool_indices_AE[insert_indices_AE[0]].unsqueeze(0), pool_indices_ROM[insert_indices_AE[0]:]))
        #     data_ROM.x = torch.cat((data_ROM.x[:insert_indices_AE[0]], torch.zeros_like(data_ROM.x[0]).unsqueeze(0), data_ROM.x[insert_indices_AE[0]:]), 0)
        
        # Insert elements into pool_indices_AE and data_AE.x
        if mask_ROM.any():
            insert_indices_ROM = diff_indices[mask_ROM]
            for idx in insert_indices_ROM:
                pool_indices_AE = torch.cat((pool_indices_AE[:idx], pool_indices_ROM[idx].unsqueeze(0), pool_indices_AE[idx:]))
                data_AE.x = torch.cat((data_AE.x[:idx], torch.zeros_like(data_AE.x[0]).unsqueeze(0), data_AE.x[idx:]), 0)
        
        # if mask_ROM.any():
        #     insert_indices_ROM = diff_indices[mask_ROM]
        #     pool_indices_AE = torch.cat((pool_indices_AE[:insert_indices_ROM[0]], pool_indices_AE[insert_indices_ROM[0]].unsqueeze(0), pool_indices_AE[insert_indices_ROM[0]:]))
        #     data_AE.x = torch.cat((data_AE.x[:insert_indices_ROM[0]], torch.zeros_like(data_AE.x[0]).unsqueeze(0), data_AE.x[insert_indices_ROM[0]:]), 0)
        
        # Calculate the difference in lengths
        len_diff = len(pool_indices_AE) - len(pool_indices_ROM)
        
        if len_diff > 0:
            # pool_indices_AE is longer, pad pool_indices_ROM and data_ROM.x
            padding = torch.zeros((len_diff, *pool_indices_ROM.shape[1:]), dtype=pool_indices_ROM.dtype, device=pool_indices_ROM.device)
            pool_indices_ROM = torch.cat((pool_indices_ROM, padding), 0)
            data_padding = torch.zeros((len_diff, *data_ROM.x.shape[1:]), dtype=data_ROM.x.dtype, device=data_ROM.x.device)
            data_ROM.x = torch.cat((data_ROM.x, data_padding), 0)
        elif len_diff < 0:
            # pool_indices_ROM is longer, pad pool_indices_AE and data_AE.x
            len_diff = -len_diff
            padding = torch.zeros((len_diff, *pool_indices_AE.shape[1:]), dtype=pool_indices_AE.dtype, device=pool_indices_AE.device)
            pool_indices_AE = torch.cat((pool_indices_AE, padding), 0)
            data_padding = torch.zeros((len_diff, *data_AE.x.shape[1:]), dtype=data_AE.x.dtype, device=data_AE.x.device)
            data_AE.x = torch.cat((data_AE.x, data_padding), 0)


       
        equilize_edges = 0

        if equilize_edges:
            i, j = 0, 0
            while i < len(pool_edges_AE) and j < len(pool_edges_ROM):
                if pool_edges_AE[i] < pool_edges_ROM[j]:
                    # Insert into pool_edges_ROM and data_ROM.edge_attr
                    pool_edges_ROM = torch.cat((pool_edges_ROM[:j], pool_edges_AE[i].unsqueeze(0), pool_edges_ROM[j:]))
                    data_ROM.edge_attr = torch.cat((data_ROM.edge_attr[:j], torch.zeros_like(data_ROM.edge_attr[0]).unsqueeze(0), data_ROM.edge_attr[j:]), 0)
                    i += 1
                elif pool_edges_AE[i] > pool_edges_ROM[j]:
                    # Insert into pool_edges_AE and data_AE.edge_attr
                    pool_edges_AE = torch.cat((pool_edges_AE[:i], pool_edges_ROM[j].unsqueeze(0), pool_edges_AE[i:]))
                    data_AE.edge_attr = torch.cat((data_AE.edge_attr[:i], torch.zeros_like(data_AE.edge_attr[0]).unsqueeze(0), data_AE.edge_attr[i:]), 0)
                    j += 1
                else:
                    i += 1
                    j += 1

            # If there are remaining elements in pool_edges_AE
            while i < len(pool_edges_AE):
                pool_edges_ROM = torch.cat((pool_edges_ROM, pool_edges_AE[i].unsqueeze(0)))
                data_ROM.edge_attr = torch.cat((data_ROM.edge_attr, torch.zeros_like(data_ROM.edge_attr[0]).unsqueeze(0)), 0)
                i += 1

            # If there are remaining elements in pool_edges_ROM
            while j < len(pool_edges_ROM):
                pool_edges_AE = torch.cat((pool_edges_AE, pool_edges_ROM[j].unsqueeze(0)))
                data_AE.edge_attr = torch.cat((data_AE.edge_attr, torch.zeros_like(data_AE.edge_attr[0]).unsqueeze(0)), 0)
                j += 1

                        # i = 0
            # while i < min(len(pool_edges_AE), len(pool_edges_ROM)):
            #     if pool_edges_AE[i] < pool_edges_ROM[i]:
            #         # Insert the element in pool_edges_ROM
            #         pool_edges_ROM = torch.cat((pool_edges_ROM[:i], pool_edges_AE[i].unsqueeze(0), pool_edges_ROM[i:]))
            #         # Assign 0 to the corresponding data.edge_attr
            #         data_ROM.edge_attr = torch.cat((data_ROM.edge_attr[:i], torch.zeros_like(data_ROM.edge_attr[0]).unsqueeze(0), data_ROM.edge_attr[i:]), 0)
            #     elif pool_edges_AE[i] > pool_edges_ROM[i]:
            #         # Insert the element in pool_edges_AE
            #         pool_edges_AE = torch.cat((pool_edges_AE[:i], pool_edges_AE[i].unsqueeze(0), pool_edges_AE[i:]))
            #         # Assign 0 to the corresponding data.edge_attr
            #         data_AE.edge_attr = torch.cat((data_AE.edge_attr[:i], torch.zeros_like(data_AE.edge_attr[0]).unsqueeze(0), data_AE.edge_attr[i:]), 0)
            #     i += 1
            
            # while len(pool_edges_AE) < len(pool_edges_ROM):
            #     pool_edges_AE = torch.cat((pool_edges_AE, torch.zeros_like(pool_edges_AE[0]).unsqueeze(0)), 0)
            #     data_AE.edge_attr = torch.cat((data_AE.edge_attr, torch.zeros_like(data_AE.edge_attr[0]).unsqueeze(0)), 0)
            # while len(pool_edges_AE) > len(pool_edges_ROM):
            #     pool_edges_ROM = torch.cat((pool_edges_ROM, torch.zeros_like(pool_edges_ROM[0]).unsqueeze(0)), 0)
            #     data_ROM.edge_attr = torch.cat((data_ROM.edge_attr, torch.zeros_like(data_ROM.edge_attr[0]).unsqueeze(0)), 0)
        
        return data_AE, data_ROM
    


    def train_ROM_simultaneous(self):
        print("Start Training ROM simultaneous")

        idf = np.random.randint(100000)

        w1 = self.w1
        w2 = self.w2
        
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0

            for i, traj_data in enumerate(self.train_data):

                traj_data = traj_data[:-1]  # Remove the last timestep from the trajectory

                traj_loader = DataLoader(traj_data, batch_size=self.batch_size)  # Create a DataLoader for the current trajectory

                node_positions= traj_data[0].pos

                # Create an iterator from traj_loader
                traj_loader_iter1 = iter(traj_loader)

                # Get the first batch of data
                first_batch = next(traj_loader_iter1)

                # Get edge_index from the first batch
                edge_index = first_batch.edge_index

                if new_model and self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index, self.batch_size)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)


                # Create an iterator from traj_loader
                traj_loader_iter = iter(traj_loader)

                trajectory_loss = 0
                trajectory_loss_AE = 0
                trajectory_loss_ROM = 0

                time_batch = 0

                # Iterate over mini-batches of time steps
                while True:
                    try:
                        data = copy.deepcopy(next(traj_loader_iter))
                    except StopIteration:
                        break  # Exit the loop if there are no more batches

                    # if self.problem == 'NS':
                    #     x_even = data.x[0::2]
                    #     x_odd = data.x[1::2]
                    #     data.x =torch.hstack((x_even, x_odd))


                    data.x += (self.noise_var) ** (0.5) * torch.randn_like(data.x)
                    u = data.x

                    # Deep copy of data since I need it twice
                    data_AE = copy.deepcopy(data)
                    
                    x_size = int(data.x.size(0) / self.batch_size)  # Calculate x_size, which is 774.0 in your case
                    time_vector = self.dt * torch.arange(self.batch_size).float().to(self.device) + time_batch  # Create time_vector
                    time_batch += self.dt * self.batch_size  # Update time_batch for the next iteration
                    rom_x = time_vector.repeat_interleave(x_size).unsqueeze(1)

                    C_x, C_y = data.center[0]
                    # print(f'Center: {C_x}, {C_y}')

                    c_x_data = C_x*torch.ones_like(data.x[:,0]).unsqueeze(1)
                    c_y_data = C_y*torch.ones_like(data.x[:,0]).unsqueeze(1)

                    data.x = torch.cat((rom_x, c_x_data, c_y_data), 1)

                    


                    # data.x = rom_x


                    # Encoder pass: Pass the current timestep to the GNN
                    data_AE, pool_indices_AE, pool_edges_AE, edge_index_AE, n_nodes_AE, ii_AE = self.net.encoder(data_AE)

                    data_ROM, pool_indices_ROM, pool_edges_ROM, edge_index_ROM, n_nodes_ROM, ii_ROM = self.net.encoder_ROM(data)

                    output_AE = self.net.decoder(data_AE, pool_indices_AE, pool_edges_AE, edge_index_AE, n_nodes_AE, ii_AE)

                    output_ROM = self.net.decoder(data_ROM, pool_indices_ROM, pool_edges_ROM, edge_index_ROM, n_nodes_ROM, ii_ROM)
                    

                    # Calculate loss as the difference between GNN output and the next timestep
                    loss_AE = ((output_AE[:,0] - u[:,0])**2).mean() 
                    loss_ROM = ((output_ROM[:,0] - u[:,0])**2).mean()
                    
                    loss = w1*loss_AE + w2*loss_ROM

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    trajectory_loss += loss.item()

                    trajectory_loss_AE += loss_AE.item()
                    trajectory_loss_ROM += loss_ROM.item()

                    total_loss += loss.item()

                trajectory_loss /= self.batch_size

                if stopwatch:
                    partial = time.time()
                    print(f"Simulation {i}: partial time: {partial-start_time:.6f} Trajectory Loss = {trajectory_loss:.6f}, AE Loss = {trajectory_loss_AE:.6f}, ROM Loss = {trajectory_loss_ROM:.6f}")

            # Scheduler step (if using LR scheduler)
            self.scheduler.step()

            # Print average loss for the epoch
            avg_loss = total_loss / (len(self.train_data)* self.batch_size)
            
            with open('loss_values.txt', 'a') as loss_file:
                loss_file.write(f"{avg_loss:.6f}\n")  # Write the loss value to the file

            print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
            end_time = time.time()
            print(f"Execution time: {end_time - start_time} seconds")

            if epoch % 10 == 0:
                torch.save(self.net.state_dict(), f'checkpoints/chk_train_ROM_{idf}_{epoch}_pool{self.pool_k}_sim_resumed_{self.resume_training}_ratiow_{w2/w1}')
                print("Saving model")

        print("End Training")
        # Save the model
        idf = np.random.randint(100000)
        torch.save(self.net.state_dict(), f'checkpoints/chk_AE_{idf}')
        print("Saving model")



                
############################################################################################################
############################################################################################################


    def forecast(self, save_plot=True, write_to_file=False):
        print("Start Evaluation")


        for i, traj_data in enumerate(self.test_data):  # Iterate over trajectories

            traj_data = traj_data[:-1]  # Remove the last timestep from the trajectory

            traj_loader = DataLoader(traj_data, batch_size=self.batch_size)  # Create a DataLoader for the current trajectory

            T, N, D = len(traj_data), traj_data[0].num_nodes, traj_data[0].num_features

            # Initialize u_t and output_t as zero tensors
            u_t = torch.zeros(T, N, D).to(self.device)
            output_t = torch.zeros_like(u_t).to(self.device)


            total_error = 0
            rel_total_error = 0

            node_positions= traj_data[0].pos
            iter1 = iter(traj_loader)
            first_batch = next(iter1)
            edge_index = first_batch.edge_index

            if new_model and self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index, self.batch_size)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)

            for t, data in enumerate(traj_loader):  # Iterate over mini-batches of time steps
                # if t==60:
                #     break
                with amp.autocast():
                    u = data.x
                    node_positions = data.pos
                    edge_index = data.edge_index
                    edge_attr = data.edge_attr
                    mesh = data.mesh

                    if self.augment_factor > 0:
                        edge_index, edge_attr = random_edge_augmentation(edge_index, edge_attr, mesh, augment_factor=self.augment_factor)

                    # Reshape u to [batch_size, N, D]
                    u_reshaped = u.view(self.batch_size, N, D)

                    # Assign u_reshaped to slices of u_t
                    for b in range(self.batch_size):
                        u_t[self.batch_size*t + b, :, :] = u_reshaped[b]

                    if self.ROM:
                        x_size = int(data.x.size(0) / self.batch_size)
                        time_vector = self.dt * torch.arange(self.batch_size).float().to(self.device) + t*self.dt*self.batch_size
                        rom_x = time_vector.repeat_interleave(x_size).unsqueeze(1)
                        data.x = rom_x
                    

                    # Forward pass with the adjusted inputs
                    output = self.net(data)

                    # Reshape output to [batch_size, N, D]
                    output_reshaped = output.view(self.batch_size, N, D)

                    # Assign output_reshaped to slices of output_t
                    for b in range(self.batch_size):
                        output_t[self.batch_size*t + b, :, :] = output_reshaped[b]

                    # Calculate reconstruction error for the current timestep
                    for b in range(self.batch_size):
                        error_t = ((output_t[self.batch_size*t+ b,:,0] - u_t[self.batch_size*t+b,:,0]) ** 2).mean().item()
                        rel_error = error_t / (u_t[self.batch_size*t+b,:,0].mean().item()**2 + 1e-10)
                        total_error += error_t
                        rel_total_error += rel_error

                if t % 5 == 0:  # Example: Clear cache every 10 iterations
                        torch.cuda.empty_cache()

            avg_error = total_error / len(traj_data)
            avg_rel_error = rel_total_error / len(traj_data)
            print(f"Test simulation {i+1}: Average Reconstruction Error {avg_error:.6f}, Average Relative Error {avg_rel_error:.6f}")

            if write_to_file:
                with open('test_errors.txt', 'a') as f:
                    f.write(f"{i+1}, {avg_error:.6f}\n")

            if save_plot:
                # Visualizing and saving outputs, ensure method is adapted for batch or single timestep visualizations
                trajectorytogif(output_t.cpu().detach(), self.dt, name=f"images/test_sim_{i}_output_{self.problem}", mesh=mesh[0])
                trajectorytogif(u_t.cpu().detach(), self.dt, name=f"images/test_sim_{i}_input_{self.problem}", mesh=mesh[0])
                # rel_square_error = (output_t - u_t)**2/(u_t**2).mean()/10 
                # trajectorytogif(rel_square_error.cpu().detach(), self.dt, name=f"images/test_sim_{i}_error_{self.problem}", mesh=mesh[0], rel_error=True) 


                # Plot pooled nodes
                pool_gif = 0
                if pool_gif:
                    plot_pooled()
          
            stop=0

        print("End Evaluation")

    # forecast_ROM 

    def forecast_ROM(self, save_plot=True, write_to_file=False):
        print("Start Evaluation ROM")


        for i, traj_data in enumerate(self.test_data):  # Iterate over trajectories

            traj_data = traj_data[:-1]  # Remove the last timestep from the trajectory

            traj_loader = DataLoader(traj_data, batch_size=self.batch_size)  # Create a DataLoader for the current trajectory

            T, N, D = len(traj_data), traj_data[0].num_nodes, traj_data[0].num_features

            # Initialize u_t and output_t as zero tensors
            u_t = torch.zeros(T, N, D).to(self.device)
            output_t = torch.zeros_like(u_t).to(self.device)


            total_error = 0

            node_positions= traj_data[0].pos
            iter1 = iter(traj_loader)
            first_batch = next(iter1)
            edge_index = first_batch.edge_index

            if new_model and self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index, self.batch_size)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)
                    self.net_ROM.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)

            for t, data in enumerate(traj_loader):  # Iterate over mini-batches of time steps
                # if t==60:
                #     break
                with amp.autocast():
                    u = data.x
                    node_positions = data.pos
                    edge_index = data.edge_index
                    edge_attr = data.edge_attr
                    mesh = data.mesh

                    if self.augment_factor > 0:
                        edge_index, edge_attr = random_edge_augmentation(edge_index, edge_attr, mesh, augment_factor=self.augment_factor)

                    # Reshape u to [batch_size, N, D]
                    u_reshaped = u.view(self.batch_size, N, D)

                    # Assign u_reshaped to slices of u_t
                    for b in range(self.batch_size):
                        u_t[self.batch_size*t + b, :, :] = u_reshaped[b]

                    x_size = int(data.x.size(0) / self.batch_size)
                    time_vector = self.dt * torch.arange(self.batch_size).float().to(self.device) + t*self.dt*self.batch_size
                    rom_x = time_vector.repeat_interleave(x_size).unsqueeze(1)
                    data.x = rom_x
                    

                    # Forward pass with the adjusted inputs
                    data, pool_indices, pool_edges, edge_index, n_nodes, ii = self.net_ROM.encoder(data)

                    output = self.net.decoder(data, pool_indices, pool_edges, edge_index, n_nodes, ii)

                    # Reshape output to [batch_size, N, D]
                    output_reshaped = output.view(self.batch_size, N, D)

                    # Assign output_reshaped to slices of output_t
                    for b in range(self.batch_size):
                        output_t[self.batch_size*t + b, :, :] = output_reshaped[b]

                    # Calculate reconstruction error for the current timestep
                    for b in range(self.batch_size):
                        error_t = ((output_t[self.batch_size*t+ b,:,0] - u_t[self.batch_size*t+b,:,0]) ** 2).mean().item()
                        total_error += error_t

                if t % 5 == 0:  # Example: Clear cache every 10 iterations
                        torch.cuda.empty_cache()

            avg_error = total_error / len(traj_data)
            print(f"Test simulation {i+1}: Average Reconstruction Error {avg_error:.6f}")

            if write_to_file:
                with open('test_errors.txt', 'a') as f:
                    f.write(f"{i+1}, {avg_error:.6f}\n")

            if save_plot:
                # Visualizing and saving outputs, ensure method is adapted for batch or single timestep visualizations
                trajectorytogif(output_t.cpu().detach(), self.dt, name=f"images/test_sim_{i}_output_{self.problem}", mesh=mesh[0])
                trajectorytogif(u_t.cpu().detach(), self.dt, name=f"images/test_sim_{i}_input_{self.problem}", mesh=mesh[0])
                # error_abs = torch.abs(output_t - u_t)
                # trajectorytogif(error_abs.cpu().detach(), self.dt, name=f"images/test_sim_{i}_error_{self.problem}", mesh=mesh[0]) 

                # Plot pooled nodes
                pool_gif = 0
                if pool_gif:
                    plot_pooled()
          
            stop=0

        print("End Evaluation")

    
    def forecast_ROM_simultaneous(self, save_plot=True, write_to_file=False):
        print("Start Evaluation ROM simultaneous")


        for i, traj_data in enumerate(self.test_data):  # Iterate over trajectories

            traj_data = traj_data[:-1]  # Remove the last timestep from the trajectory

            traj_loader = DataLoader(traj_data, batch_size=self.batch_size)  # Create a DataLoader for the current trajectory

            T, N, D = len(traj_data), traj_data[0].num_nodes, traj_data[0].num_features

            # Initialize u_t and output_t as zero tensors
            u_t = torch.zeros(T, N, D).to(self.device)
            output_t = torch.zeros_like(u_t).to(self.device)


            total_error = 0

            node_positions= traj_data[0].pos
            iter1 = iter(traj_loader)
            first_batch = next(iter1)
            edge_index = first_batch.edge_index

            if new_model and self.multi_scaling:
                    # Maps creation for performance, assuming similar need as in training
                    fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list = create_maps_distances(node_positions, self.scales, edge_index, self.batch_size)
                    self.net.assign_maps_coarsening(fine2coarse_list, distances_list, edge_index_coarse_list, fine2coarse_edge_list, node_coarse_list)

            for t, data in enumerate(traj_loader):  # Iterate over mini-batches of time steps
                # if t==60:
                #     break
                with amp.autocast():
                    u = data.x
                    node_positions = data.pos
                    edge_index = data.edge_index
                    edge_attr = data.edge_attr
                    mesh = data.mesh

                    if self.augment_factor > 0:
                        edge_index, edge_attr = random_edge_augmentation(edge_index, edge_attr, mesh, augment_factor=self.augment_factor)

                    # Reshape u to [batch_size, N, D]
                    u_reshaped = u.view(self.batch_size, N, D)

                    # Assign u_reshaped to slices of u_t
                    for b in range(self.batch_size):
                        u_t[self.batch_size*t + b, :, :] = u_reshaped[b]

                    
                    # Calculate x_size
                    x_size = int(data.x.size(0) / self.batch_size)
                    
                    # Create time_vector
                    time_vector = self.dt * torch.arange(self.batch_size).float().to(self.device) + t*self.dt*self.batch_size
                    
                    # Update time_batch for the next iteration
                    # time_batch += self.dt * self.batch_size
                    
                    # Repeat time_vector to match x_size and reshape it
                    rom_x = time_vector.repeat_interleave(x_size).unsqueeze(1)
                    
                    # Extract C_x and C_y from data.center
                    C_x, C_y = data.center[0]
                    
                    # Create tensors filled with C_x and C_y
                    c_x_data = C_x * torch.ones_like(data.x[:, 0]).unsqueeze(1)
                    c_y_data = C_y * torch.ones_like(data.x[:, 0]).unsqueeze(1)
                    
                    # Concatenate rom_x, c_x_data, and c_y_data to form the new data.x
                    data.x = torch.cat((rom_x, c_x_data, c_y_data), 1)
                    

                    # Forward pass with the adjusted inputs
                    data, pool_indices, pool_edges, edge_index, n_nodes, ii = self.net.encoder_ROM(data)

                    output = self.net.decoder(data, pool_indices, pool_edges, edge_index, n_nodes, ii)

                    # Reshape output to [batch_size, N, D]
                    output_reshaped = output.view(self.batch_size, N, D)

                    # Assign output_reshaped to slices of output_t
                    for b in range(self.batch_size):
                        output_t[self.batch_size*t + b, :, :] = output_reshaped[b]

                    # Calculate reconstruction error for the current timestep
                    for b in range(self.batch_size):
                        error_t = ((output_t[self.batch_size*t+ b,:,0] - u_t[self.batch_size*t+b,:,0]) ** 2).mean().item()
                        total_error += error_t

                if t % 5 == 0:  # Example: Clear cache every 10 iterations
                        torch.cuda.empty_cache()

            avg_error = total_error / len(traj_data)
            print(f"Test simulation {i+1}: Average Reconstruction Error {avg_error:.6f}")

            if write_to_file:
                with open('test_errors.txt', 'a') as f:
                    f.write(f"{i+1}, {avg_error:.6f}\n")

            if save_plot:
                # Visualizing and saving outputs, ensure method is adapted for batch or single timestep visualizations
                trajectorytogif(output_t.cpu().detach(), self.dt, name=f"images/test_sim_{i}_output_{self.problem}", mesh=mesh[0])
                trajectorytogif(u_t.cpu().detach(), self.dt, name=f"images/test_sim_{i}_input_{self.problem}", mesh=mesh[0])

                # Plot pooled nodes
                pool_gif = 0
                if pool_gif:
                    plot_pooled()
          
            stop=0

        print("End Evaluation")




    def test_checkpoints(self):
        
        print("Start testing checkpoints")

        with open('test_errors.txt', 'w') as f:
            f.write("Simulation, Average Reconstruction Error\n")
        for i in range(19):
            # if i !=4:
            ii=i*10
            # self.net.load_state_dict(torch.load(f'from_hpc/checkpoints_stokes/chk_AE_10137_{ii}', map_location = self.device))
            self.net.load_state_dict(torch.load(f'checkpoints/chk_ENC_DEC_romFalse_86364_{ii}_pool0.5_sim_resumed_False_batch_size_25', map_location = self.device))
            # self.net.load_state_dict(torch.load(f'from_hpc/chk_AE__{ii}', map_location = self.device))
            self.forecast(save_plot=False, write_to_file=True)
            # self.forecast_ROM_simultaneous(save_plot=False, write_to_file=True)

        print("End testing checkpoints")





def noise_wake2(output_t, noise=1e-3, threshold=1.3, threshold2=1.2, threshold_low=0.01, intense_noise_factor=10):
    """
    Introduces intense noise in the region of the solution around the obstacle.
    
    Parameters:
    - output_t (torch.Tensor): The solution tensor.
    - noise (float): The base noise level.
    - threshold (float): The threshold to identify the region around the obstacle.
    - intense_noise_factor (float): The factor by which the noise is intensified around the obstacle.
    
    Returns:
    - torch.Tensor: The solution tensor with added noise.
    """

    # mask_change

    base_noise = noise * torch.randn_like(output_t)
    # Generate intense noise with more variability
    intense2_noise = intense_noise_factor/2 * noise * torch.randn_like(output_t) + noise*0.8*torch.ones_like(output_t)
    intense_noise = intense_noise_factor * noise * torch.randn_like(output_t) + noise*torch.ones_like(output_t)

    mask2 = output_t > threshold2
    
    # Create a mask for regions where the solution is greater than the threshold
    mask = torch.logical_or(output_t > threshold, output_t < threshold_low)

    output_t = torch.where(mask2, output_t + intense2_noise, output_t)

    
    # Apply intense noise where the mask is True
    output_t = torch.where(mask, output_t + intense_noise, output_t + base_noise)

    output_t = output_t+base_noise

    
    
    return output_t


def noise_wake(output_t, noise=1e-4, ratios=[0.4, 0.05, 0.035, 0.01], intensities=[0.005, 100, 110, 120]):
    """
    Introduces three levels of intense noise to points with the greatest changes with respect to the previous timestep.
    
    Parameters:
    - output_t (torch.Tensor): The solution tensor of shape (time_steps, points, 1).
    - noise (float): The base noise level.
    - ratios (list of float): The percentages of points to retain for each noise level.
    - intensities (list of float): The intensities of noise for each level.
    
    Returns:
    - torch.Tensor: The solution tensor with added noise.
    """
    # Iterate over time steps starting from the second one
    for t in range(1, output_t.shape[0]):
        # Compute the absolute difference between the current and previous time step
        diff = torch.abs(output_t[t] - output_t[t-1])
        
        # Flatten the differences
        diff_flat = diff.view(-1)
        
        # Apply each level of noise
        for ratio, intensity in zip(ratios, intensities):
            # Number of points to retain for this level
            num_points_to_retain = int(output_t.shape[1] * ratio)
            
            # Get the indices of the top percentage of differences
            _, top_indices = torch.topk(diff_flat, num_points_to_retain)
            
            # Create a mask for the selected nodes
            mask = torch.zeros_like(diff_flat, dtype=torch.bool)
            mask[top_indices] = True
            mask = mask.view(diff.shape)
            
            # Generate intense noise for this level
            intense_noise = intensity * noise * torch.randn_like(output_t[t]) + noise * 10 * torch.ones_like(output_t[t])
            
            # Apply intense noise where the mask is True
            output_t[t] = torch.where(mask, output_t[t] + intense_noise, output_t[t])
    
    return output_t


def add_crescent_numbers(output_t):
    """
    Adds a crescent number to each node in order for each timestep.
    
    Parameters:
    - output_t (torch.Tensor): The solution tensor of shape (time_steps, points, 1).
    
    Returns:
    - torch.Tensor: The solution tensor with added crescent numbers.
    """
    # Iterate over time steps
    for t in range(output_t.shape[0]):
        # Iterate over nodes within each time step
        for i in range(300,800):
            # Add the node index to each element
            if i%100>70 and i%100<90:
                output_t[t, i, 0] += i*100
    
    return output_t




def error_t(output_t, deltat, factor):
    """
    Returns output increased with the difference between output_t at each timestep and output_t at timestep + deltat.
    
    Parameters:
    - output_t (torch.Tensor): The solution tensor of shape (time_steps, points, 1).
    - deltat (int): The timestep difference to compute the error.
    
    Returns:
    - torch.Tensor: The solution tensor with added differences.
    """
    diff = torch.zeros_like(output_t)

    # Iterate over time steps up to the length of output_t minus deltat
    for t in range(output_t.shape[0] - deltat):
        # Compute the difference between output_t at the current timestep and output_t at timestep + deltat
        diff[t] = (output_t[t + deltat] - output_t[t])*factor
    
    return diff

