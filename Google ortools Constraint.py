from ortools.sat.python import cp_model

#ダイレクトに答え一つ出す
def SimpleSatProgram():
    #"Creates the model"
    model = cp_model.CpModel()

    #Creates the variables
    num_vals = 3
    x = model.NewIntVar(0,num_vals - 1,"x")
    y = model.NewIntVar(0,num_vals - 1,"y")
    z = model.NewIntVar(0,num_vals - 1,"z")

    #Creates the constraints（制約)
    model.Add(x != y)

    #Creates a solver and solves the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.FEASIBLE:
        print("x = %i" % solver.Value(x))
        print("y = %i" % solver.Value(y))
        print("z = %i" % solver.Value(z))

#全部出す(上記後となる。)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ortools.sat.python import cp_model


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()

    def solution_count(self):
        return self.__solution_count


def SearchForAllSolutionsSampleSat():
    # Creates the model.
    model = cp_model.CpModel()

    # Creates the variables.
    num_vals = 3
    x = model.NewIntVar(0, num_vals - 1, 'x')
    y = model.NewIntVar(0, num_vals - 1, 'y')
    z = model.NewIntVar(0, num_vals - 1, 'z')

    # Create the constraints.
    model.Add(x != y)

    # Create a solver and solve.
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter([x, y, z])
    status = solver.SearchForAllSolutions(model, solution_printer)

    print('Status = %s' % solver.StatusName(status))
    print('Number of solutions found: %i' % solution_printer.solution_count())


SearchForAllSolutionsSampleSat()
