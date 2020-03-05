# pip install ortools
# from ortools.linear_solver import pywraplp ←指定された~とか出たら、　pip install --upgrade ortools==?? --user

#Get started Guide
#線形計画法(linear # OPTIMIZE: )
# Create the variables x and y.
from __future__ import print_function
from ortools.linear_solver import pywraplp


def main():
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver('simple_lp_program',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING) #solverでどのようなことをやるか指定

    # Create the variables x and y.
    x = solver.NumVar(0, 1, 'x') #0 <= x <= 1
    y = solver.NumVar(0, 2, 'y') #0 <= y <= 2

    print('Number of variables =', solver.NumVariables()) #変数の数

    # Create a linear constraint, 0 <= x + y <= 2.
    ct = solver.Constraint(0, 2, 'ct')
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print('Number of constraints =', solver.NumConstraints()) #条件は1つ

    # Create the objective function, 3 * x + y.
    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 1)
    objective.SetMaximization()

    solver.Solve()

    print('Solution:')
    print('Objective value =', objective.Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())


if __name__ == '__main__': #http://azuuun-memorandum.hatenablog.com/entry/2015/05/09/002549
    main()


#linear practice
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('simple_lp_program',
                         pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#create variables
x = solver.NumVar(0,solver.infinity(),"x")
y = solver.NumVar(0,solver.infinity(),"y")

#create constraint(制約)
#constraint 1
constraint0 = solver.Constraint(-solver.infinity(), 14)
constraint0.SetCoefficient(x, 1)
constraint0.SetCoefficient(y,2)

#Constraint 2
constraint1 = solver.Constraint(0,solver.infinity())
constraint1.SetCoefficient(x,3)
constraint1.SetCoefficient(y, -1)

#Constraint 3
constraint2 = solver.Constraint(-solver.infinity(),2)
constraint2.SetCoefficient(x,1)
constraint2.SetCoefficient(y, -1)

#Define object function
# Objective function: 3x + 4y.
objective = solver.Objective()
objective.SetCoefficient(x, 3)
objective.SetCoefficient(y, 4)
objective.SetMaximization()

    # Solve the system.
solver.Solve() #解く
opt_solution = 3 * x.solution_value() + 4 * y.solution_value()
print('Number of variables =', solver.NumVariables())
print('Number of constraints =', solver.NumConstraints())
# The value of each variable in the solution.
print('Solution:')
print('x = ', x.solution_value()) #x ,y ,の値
print('y = ', y.solution_value())
# The objective value of the solution.
print('Optimal objective value =', opt_solution) #結果


#python_practiceへ
df = pd.read_csv("some.csv",header = None) #https://note.nkmk.me/python-pandas-read-csv-tsv/
food = [[]] * len(df)
#minimize the sum of price of foods
solver = pywraplp.Solver('SolveStigler',
                           pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
objective = solver.Objective()
for i in range(0,len(df)):
    food[i] = solver.NumVar(0.0,solver.infinity(),df[0][i]) #food[i]が変数
    objective.SetCoefficient(food[i],1) #変数羅列
objective.SetMinimization()

#create tje constraints one per nutrient
constraints = [0]*len(nutrients)
for i in range(0,len(nutrients)):
    constraints[i] = solver.Constraint(nutrients[i][1],solver.infinity())#各iごとの制約
    for j in range(0,len(df)):
        constraints[i].SetCoefficient(food[j],data[j][i+3]) #各変数の制約 i固定j移動

status = solver.Solve()

if status == solver.OPTIMAL:
  # Display the amounts (in dollars) to purchase of each food.
  price = 0
  num_nutrients = len(data[i]) - 3 # 頭3つはnutに無関係
  nutrients = [0] * (len(data[i]) - 3)
  for i in range(0, len(data)):
    price += food[i].solution_value()

    for nutrient in range(0, num_nutrients):
      nutrients[nutrient] += df[nutrient+3][i] * food[i].solution_value()

    if food[i].solution_value() > 0:
      print('%s = %f' % (data[i][0], food[i].solution_value()))

  print('Optimal annual price: $%.2f' % (365 * price))
else:  # No optimal solution was found.
  if status == solver.FEASIBLE:
    print('A potentially suboptimal solution was found.')
  else:
    print('The solver could not solve the problem.')
