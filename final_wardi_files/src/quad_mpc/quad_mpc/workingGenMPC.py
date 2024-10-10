from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
from .workingModel import Quadrotor
import importlib
import sys

class QuadrotorMPC2:
    def __init__(self, generate_c_code: bool, quadrotor: Quadrotor, horizon: float, num_steps: int):
        
        self.model = AcadosModel() # instantiate empty class AcadosModel
        self.quad = quadrotor # the quadrotor model from the workingModel.py file
        self.model_name = 'holybro' # name of the model
        self.horizon = horizon # MPC horizon length in seconds
        self.num_steps = num_steps # number of discretization steps in the horizon

        self.ocp_solver = None # initialize the ocp solver to None so that it can be set later with compiled cython MPC code
        self.generate_c_code = generate_c_code # if True, will generate and compile fresh acados mpc even if it exists
        self.code_export_directory = str(self.model_name) + '_mpc' + '_c_generated_code' # where to look for the compiled c code or store if generating fresh
        
        if self.generate_c_code:
            print("\n\n\nchosen to generate and compile fresh acados mpc even if it exists... \n\n\n")
            self.generate_mpc()
            print("done compiling fresh acados mpc!")
        else:
            try:
                print("\n\n\ntrying to find compiled acados mpc. if not found, will generate and compile a fresh one\n\n\n")
                sys.path.append(self.code_export_directory) # add the code export directory to the path so that the compiled c code can be found if it exists
                acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx') # import the compiled c code in the pyx file
                self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps) # set the ocp solver to the compiled c code!

            except ImportError:
                print("compiled acados mpc not found. generating and compiling fresh... ")
                self.generate_mpc()
                print("done compiling fresh acados mpc!")


    def generate_mpc(self):
        f_expl, x, u = self.quad.dynamics()
        
        # Define Acados Model
        model = AcadosModel()   
        model.f_expl_expr = f_expl
        model.x = x
        model.u = u
        model.name = self.model_name

        # Define Acados OCP
        ocp = AcadosOcp() # instantiate empty class AcadosOcp
        ocp.model = model # give it the model
        ocp.code_export_directory = self.code_export_directory # set the code export directory

        # Define important parameters
        nx = model.x.size()[0] # number of states from the model
        nu = model.u.size()[0] # number of controls from the model
        Tf = self.horizon # MPC horizon length in seconds
        N = self.num_steps # number of discretization steps in the horizon

        # Define the temporal qualities of the MPC
        ocp.dims.N = N # set N in the ocp object
        ocp.solver_options.tf = Tf # set Tf in the ocp object

        # All the constraints
        ## bounds on control
        max_rate = 0.8
        max_thrust = 27.0
        min_thrust = 0.0
        ocp.constraints.lbu = np.array([min_thrust, -max_rate, -max_rate, -max_rate])
        ocp.constraints.ubu = np.array([max_thrust, max_rate, max_rate, max_rate])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ## initial state constraint (placeholder x0 because it is set in real time in the loop)
        ocp.constraints.x0 = np.zeros(9)


        # Define the cost function

        ## For Linear Least Squares cost: 
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.cost.Vx = np.diag([1.,1.,1.,1.,1.,1.,1.,1.,1.]) # cost weight matrix for states - identity matrix basically but by writing it like this I can experiment more easily
        ocp.cost.Vu = np.zeros((9, 4)) # cost weight matrix for controls - zero matrix to simplify the problem to quadratic in states error only
# Linear_LS cost eqn after Vx = eye and Vu = 0:
    # Linear_LS = .5 || Vx*x + Vu*u - y_ref ||_{W}^2 = .5 || x - y_ref ||_{W}^2 = .5 * ||y_err||_{W}^2 = .5 * y_err^T * W * y_err -> quadratic in y_err only!!
        ocp.cost.W = np.diag([50., 50., 50., 10., 10., 10., 5., 5., 5.]) #[x, y, z, vx, vy, vz, roll, pitch, yaw] - cost weight matrix for states error
        ocp.cost.yref = np.zeros(9)
        ocp.model.cost_expr_ext_cost_e = 0.

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP'


        ## Generation, compilation, and loading of the acados solver

        # create ocp solver: json files are where solvers are stored and loaded from in the non-compiled case. the json file is also the base for compilation
        json_file = str(self.model_name) + '_mpc' + '_acados_ocp.json' # name of the json file to store the ocp
        AcadosOcpSolver.generate(ocp, json_file= json_file)  # generate the ocp and store it in the json file. note if you comment this out, compilation will fail :)

        # compile the ocp solver (generated above and stored in the json file) with cython in the code export directory
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True) # after generating the json file, compile all of it with cython and store it in the code export directory
        
        # import the compiled c code and set the ocp solver to it
        sys.path.append(ocp.code_export_directory) # add the code export directory to the path so that the compiled c code can be found now that it exists
        acados_ocp_solver_pyx = importlib.import_module('acados_ocp_solver_pyx') # import the compiled c code in the pyx file
        self.ocp_solver = acados_ocp_solver_pyx.AcadosOcpSolverCython(self.model_name, 'SQP', self.num_steps) # set the ocp solver to the compiled c code!

    def solve_mpc_control(self, x0, xd):
        N = self.num_steps # number of discretization steps in the horizon

        if xd.shape[1] != N: # raise error if the reference trajectory doesn't have the the right number of steps
            raise ValueError("The reference trajectory should have the same length as the number of steps")
        
        for i in range(N): # set the reference trajectory in the solver for each of the N steps
            y_ref = xd[:, i] #yref = xd for each step
            self.ocp_solver.set(i, 'y_ref', y_ref) # set it in the solver

        # set initial state
        self.ocp_solver.set(0, 'lbx', x0) # at index 0 (initial state) set lower bound of x to x0
        self.ocp_solver.set(0, 'ubx', x0) # at index 0 (initial state) set upper bound of x to x0

        status = self.ocp_solver.solve() # solve the optimization problem
        x_mpc = self.ocp_solver.get(0, 'x') # get the optimal state trajectory
        u_mpc = self.ocp_solver.get(0, 'u') # get the optimal control trajectory

        return status, x_mpc, u_mpc

