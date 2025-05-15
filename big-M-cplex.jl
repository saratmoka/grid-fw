using JuMP, CPLEX
using DelimitedFiles
using LinearAlgebra
cd(@__DIR__)


#Specify the path to the dataset. Uncomment only one of three options

# file = "./data/example1/31assets.csv"
file = "./data/example1/85assets.csv"
# file = "./data/example1/89assets.csv"
# file = "./data/example1/98assets.csv"
# file = "./data/example1/225assets.csv"
# file = "./data/example2/2000-2007.csv"
# file = "./data/example2/2008-2015.csv"
# file = "./data/example2/2016-2023.csv"
# file = "./data/example3/1000assets.csv"
# file = "./data/example3/2000assets.csv"



# Load the dataset as (covariance) matrix

Σ = readdlm(file, ',', Float64)
p = size(Σ, 1)


# Specify the model size k and time limit (in secs) for termination
k = 4
time_limit = 60


# Specify value of M in the Big-M method
M = 1.0  # Upper bound for asset weights



# Create model
model = Model(CPLEX.Optimizer)


# Uncomment the following line to force single‐threaded solves
#set_optimizer_attribute(model, "CPX_PARAM_THREADS", 1)


# Spacify the constraints

# Decision variables
@variable(model, -M <= w[i=1:p] <= M)   # Portfolio weights
@variable(model, z[i=1:p], Bin)        # Binary selection variables

# Objective: Minimize portfolio variance
@objective(model, Min, sum(w[i] * Σ[i,j] * w[j] for i in 1:p, j in 1:p))

# Constraints
@constraint(model, sum(w) == 1)                 # Portfolio budget constraint
@constraint(model, sum(z) <= k)                 # Cardinality constraint (max k assets)
@constraint(model, [i=1:p], w[i] <= M * z[i])   # Linking upper constraint
@constraint(model, [i=1:p], w[i] >= -M * z[i])  # Linking lower constraint


# Set the time limit in secs (terminate when it reaches this time)
set_optimizer_attribute(model, "CPX_PARAM_TILIM", time_limit)


# Solve the problem using CPLEX
optimize!(model)


# Print results
if termination_status(model) == MOI.TIME_LIMIT
    println("Terminated due to time limit. Best solution found:")
    println("Selected assets: ", [i for i in 1:p if value(z[i]) > 0.5])
    println("Portfolio weights: ", [value(w[i]) for i in 1:p])
    println("Minimum variance: ", objective_value(model))
elseif termination_status(model) == MOI.OPTIMAL
    println("Terminated on optimality")
    println("Selected assets: ", [i for i in 1:p if value(z[i]) > 0.5])
    println("Portfolio weights: ", [value(w[i]) for i in 1:p])
    println("Minimum variance: ", objective_value(model))
else
    println("Terminated for another reason (i.e., neither time limit nor optimality)")
    println("Selected assets: ", [i for i in 1:p if value(z[i]) > 0.5])
    println("Portfolio weights: ", [value(w[i]) for i in 1:p])
    println("Minimum variance: ", objective_value(model))
end