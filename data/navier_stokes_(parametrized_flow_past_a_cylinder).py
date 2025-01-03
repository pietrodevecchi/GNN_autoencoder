# -*- coding: utf-8 -*-
"""Navier-Stokes (parametrized flow past a cylinder).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rtpPKI9zy9o0QnLjQAlKpL5Z4kDcxISX
"""
import numpy as np
import sys
import subprocess

# try:
from dlroms import *
# except ImportError:
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/NicolaRFranco/dlroms.git'])
#     from dlroms import*

from fenics import *
import numpy as np
from IPython.display import clear_output as clc
from dlroms import num2p
import dlroms.fespaces as fe

import matplotlib.pyplot as plt
import numpy as np
import fenics as fe

from utils import asfunction

import time


import numpy as np

def circle(xc, yc, r, h = None):
  if(h is None):
    h = r/100.0
  nh = int(2*np.pi*r/h)
  ths = np.linspace(0, 2*np.pi, nh)
  return fe.polygon([(xc+r*np.cos(th), yc+r*np.sin(th)) for th in ths])

def channel(length, width, fine_till, H, h):
  if(h is None):
    h = width/100.0
    H = 10*h
  points = [[x, 0] for x in np.linspace(0, fine_till, int(fine_till/h)+1)]
  points += [[x, 0] for x in np.linspace(fine_till+H, length, int((length-fine_till)/H))]
  points += [[length, y] for y in np.linspace(H, width, int(width/H))]
  points += [[x, width] for x in np.flip(np.linspace(fine_till+H, length-H, int((length-fine_till)/H)-1))]
  points += [[x, width] for x in np.flip(np.linspace(0, fine_till, int(fine_till/h)+1))]
  points += [[0, y] for y in np.flip(np.linspace(0, width-h, int(width/h)))]
  return fe.polygon(points)
  # return fe.Polygon(points)

# change working directory

import os
os.chdir('data/NS/')

channel_length = 2.2
channel_width = 0.41





"""# PDE solution

The following code returns the velocity magnitude for $t\in[0,T]$. At every time-step, the solution is stored as a vector of nodal values, listed according to fenics' sorting for P1 Lagrangian elements (see $\texttt{fe.coordinates(Vm)}$).
"""

# Time discretization
T = 3.5
num_steps = int(1000*T)
dt = T / num_steps

data={}
start = time.time()

for i in range(1, 100):

  name = "geometry" + str(i)


  H = 0.05 # Large stepsize to be used faraway from the obstacle
  h = 0.025 # Fine stepsize, used nearby the obstacle and the inflow

  # center of the circle moves in the square of bottom left 
  # vertex in (0.1,0.1) and side 0.2
  circle_center = 1.6*(i%10)/100+0.12, 1.6*(i//10)/100+0.12
  print(i)
  print(circle_center)
  print("partial time: ", time.time()-start)
  # circle_center = 0.2, 0.2+0.01


  circle_radius = 0.05

  domain = channel(channel_length, channel_width, fine_till = 3*circle_center[0], H = H, h = h)
  domain = domain - circle(*circle_center, circle_radius, h = h)

  domain_mesh = fe.mesh(domain, stepsize = H)

  clc()
  print(i)
  print(circle_center)
  print("partial time: ", time.time()-start)

  # right mesh
  # fe.savemesh(name + ".xml", domain_mesh)

  # fe.plot(domain_mesh)

  # fig = fe.plot(domain_mesh)

  # Define function spaces
  V = VectorFunctionSpace(domain_mesh, 'P', 2)
  Q = FunctionSpace(domain_mesh, 'P', 1)
  Vm = fe.space(domain_mesh, 'CG', 2)

  
  # right mesh
  fe.savemesh(name + ".xml", Vm.mesh)


  # Problem data
  inflow   = 'near(x[0], 0)'
  outflow  = 'near(x[0], %.2f)' % channel_length
  walls    = 'near(x[1], 0) || near(x[1], %.2f)' % channel_width
  xl, xr = circle_center[0]-1.2*circle_radius, circle_center[0]+1.2*circle_radius
  yl, yr = circle_center[1]-1.2*circle_radius, circle_center[1]+1.2*circle_radius
  cylinder = 'on_boundary && x[0]>%.2f && x[0]<%.2f && x[1]>%.2f && x[1]<%.2f' % (xl, xr, yl, yr)

  inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

  bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
  bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
  bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
  bcp_outflow = DirichletBC(Q, Constant(0), outflow)
  bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
  bcp = [bcp_outflow]

  # clc()

  def FOMsolver(mu, rho, verbose = False):
      # Trial and test functions
      u = TrialFunction(V)
      v = TestFunction(V)
      p = TrialFunction(Q)
      q = TestFunction(Q)

      # Define functions for solutions at previous and current time steps
      u_n = Function(V)
      u_  = Function(V)
      p_n = Function(Q)
      p_  = Function(Q)

      # Define expressions used in variational forms
      U  = 0.5*(u_n + u)
      n  = FacetNormal(domain_mesh)
      f  = Constant((0, 0))
      k  = Constant(dt)
      mu = Constant(mu)
      rho = Constant(rho)

      # Define symmetric gradient
      def epsilon(u):
          return sym(nabla_grad(u))

      # Define stress tensor
      def sigma(u, p):
          return 2*mu*epsilon(u) - p*Identity(len(u))

      # Define variational problem for step 1
      F1 = (  rho*dot((u - u_n) / k, v)*dx
            + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx
            + inner(sigma(U, p_n), epsilon(v))*dx
            + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds
            - dot(f, v)*dx)
      a1 = lhs(F1)
      L1 = rhs(F1)

      # Define variational problem for step 2
      a2 = dot(nabla_grad(p), nabla_grad(q))*dx
      L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

      # Define variational problem for step 3
      a3 = dot(u, v)*dx
      L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

      # Assemble matrices
      A1 = assemble(a1)
      A2 = assemble(a2)
      A3 = assemble(a3)

      # Apply boundary conditions to matrices
      [bc.apply(A1) for bc in bcu]
      [bc.apply(A2) for bc in bcp]

      solution = []

      # Time-stepping
      t = 0
      solution.append(u_.vector()[:]+0.0)
      for n in range(num_steps):
          if(n%10==0 and verbose):
              clc(wait = True)
              print("Progress: %s." % num2p(n/num_steps))
          # Update current time
          t += dt

          # Step 1: Tentative velocity step
          b1 = assemble(L1)
          [bc.apply(b1) for bc in bcu]
          solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

          # Step 2: Pressure correction step
          b2 = assemble(L2)
          [bc.apply(b2) for bc in bcp]
          solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

          # Step 3: Velocity correction step
          b3 = assemble(L3)
          solve(A3, u_.vector(), b3, 'cg', 'sor')


          # Update previous solution
          u_n.assign(u_)
          p_n.assign(p_)
          solution.append(u_.vector()[:]+0.0)
      if(verbose):
          clc()
      sol2 = [solution[i] for i in range(0, num_steps+1, 25)]
      umod = np.linalg.norm(np.stack(sol2).reshape(len(sol2), -1, 2), axis = -1)

      # Print size of solution
      print("Size of solution (umod.shape):", umod.shape)
      
      # Print size of solution at time 0
      print("Size of solution at time 0 (umod[0].shape):", umod[0].shape)
      
      # Print mesh information (number of cells and dof)
      print("Number of cells in the mesh (domain_mesh.num_cells()):", domain_mesh.num_cells())
      print("Number of vertices in the mesh (domain_mesh.num_vertices()):", domain_mesh.num_vertices())
      print("Mesh information (domain_mesh):", domain_mesh)
      print("Function space (Vm):", Vm)
      print("Dimension of function space (Vm.dim()):", Vm.dim())
      print("DOF coordinates (Vm.tabulate_dof_coordinates()):", Vm.tabulate_dof_coordinates())
      print("Shape of DOF coordinates (Vm.tabulate_dof_coordinates().shape):", Vm.tabulate_dof_coordinates().shape)
      
      # Inspect type and dimension of u
      print("Type of umod:", type(umod))
      print("Shape of umod:", umod.shape)
      
      # Assuming umod, mesh, asfunction, and other necessary variables are defined elsewhere in the script

      mesh = Vm.mesh
      
      # Plotting the umod in a gif
      

      return umod



  # # def visualize(umod):
  # #     from IPython.display import Image
  # #     fig = fe.animate(umod[::2], Vm, figsize=(8, 2))
  # #     # Save the figure to a temporary file and return it as an Image object
  # #     fig.savefig('/tmp/animation.png')
  # #     return Image(filename='/tmp/animation.png')


  # u = FOMsolver(mu = 0.001, rho = 0.75) # Typically takes around 3-7 minutes on Colab, depending on the discretization

  # # visualize velocity u 

  # fe.gif(u.gif, u, Vm, figsize=(8, 2))

  # inspect u

  # print(u.shape)
  # print(u[0].shape)

  if(i//10==0) and False:
    fe.gif('u.gif', u, Vm, dt=dt*25, T=T, figsize=(8, 2), colorbar=True)
  # fe.gif('u.gif', u, Vm, dt=dt*25, T=T, figsize=(8, 2), colorbar=True)

  if False:
    data[i] = {"mesh": name ,"umod":u.tolist()}

  # # save solution to json file

if False:
  import json
  with open('data.json', 'w') as f:
      json.dump(data, f)



