from source.bus_route_data_processor import SchoolBusRouteDataBuilder
from data.bus_router_model import SchoolBusRouter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math, random
import copy, time

# Sample case is information provided by Auckland, New Zealand
bus_size = 48
auckland_latitude = -36.848461
bus_stop_filename = "source/School_Bus_Stop.csv"
school_filename = "source/Directory-School-Current-Data.csv"

stop_file = "data/stop-data.csv"
school_file = "data/school-data.csv"

#Initialization of data
if(not (Path(stop_file).is_file() and Path(school_file).is_file())):
    data_source = SchoolBusRouteDataBuilder(auckland_latitude)
    if(not data_source.buildSampleData(bus_stop_filename, school_filename)):
        print("Error when building a the dataset")

bus_router = SchoolBusRouter(stop_file, school_file, bus_size)
#Get all coordinates & school names
coordinates, schools = bus_router.prepare()
school_keys = list(schools.keys())


def Find_Stop_Index(array, target):
    for x in range(len(array)):
        arr = array[x]
        for y in range(len(arr)):
            point = arr[y]
            if(point[0] == target[0] and point[1] == target[1]):
                return [x, y]
    return [-1, -1]

def Route_Distance(route):
    if(len(route) <= 1):
        return 0
    total = 0
    for x in range(len(route)-1):
        pointA = route[x]
        pointB = route[x+1]
        distance = math.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 )
        total += distance
    return total

def Draw_Bus_Routes(school_name, school_routes):
    data, goal = bus_router.sample(school_name)

    # Put the coordinates & students in each stop into an array
    points = np.array(list(data.keys()))
    students = np.fromiter(data.values(), dtype=int)

    # Visualize the situation in a scatterplot
    X = np.append(points[:,0], goal[0])
    Y = np.append(points[:,1], goal[1])
    S = np.append(students, 50)
    c = np.append([0 for x in range(len(X)-1)], 250)
    plt.clf()
    plt.scatter(X, Y, S, c=c)

    for route in school_routes:
        for x in range(len(route)-1):
            pointA = route[x]
            pointB = route[x+1]

            x_line = [pointA[0], pointB[0]]
            y_line = [pointA[1], pointB[1]]
            plt.plot(x_line, y_line, "ro-", markersize=0)

    plt.xlabel("Latitude (°)")
    plt.ylabel("Longitude (°)")
    plt.show()
    

def Best_First_Search(school_name):
    data, goal = bus_router.sample(school_name)
    school_routes = []

    # Put the coordinates & students in each stop into an array
    points = np.array(list(data.keys()))
    students = np.fromiter(data.values(), dtype=int)

    school_routes = []
    while(sum(students) > 0):
        # Find most distant node
        distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))
        sorted_indices = np.argsort(distances)
        distant_point = points[sorted_indices[-1]]

        # Start a bus route from there
        space = bus_size - students[sorted_indices[-1]]
        points = np.delete(points, sorted_indices[-1], 0)
        students = np.delete(students, sorted_indices[-1], 0)

        # Generate a sublist of neighbour nodes
        #Distance between 'neighbour' to distant node does not exceed distance between distant & school
        max_N_dist = distances[sorted_indices[-1]]
        N_dist = np.sqrt(np.square(distant_point[0]-points[:,0]) + np.square(distant_point[1]-points[:,1]))

        # Remove distant nodes & nodes that have too many students
        # Get indices, the point, students on point, and distance from goal
        N_ind = np.where( (N_dist <= max_N_dist) & (students <= space) )[0]
        N = points[N_ind]
        N_stud = students[N_ind]
        goal_dist = np.sqrt(np.square(goal[0]-N[:,0]) + np.square(goal[1]-N[:,1]))

        curr = distant_point
        path = []
        # Add routes if there's space in the bus
        while(space > 0):
            # Compute heuristic potential
            dists = np.sqrt(np.square(curr[0]-N[:,0]) + np.square(curr[1]-N[:,1]))
            potential = dists/(1 + N_stud) + goal_dist/(1 + space)
            potential[N_stud > space] = 0
            potential[path] = 0 # Tabu: Do not revisit previous nodes
            if(sum(potential) == 0):
                break

            # Visit most potential node
            sort_pot = np.argsort(potential)
            target_ind = sort_pot[-1]
            
            path.append(target_ind)
            curr = N[target_ind]
            space -= N_stud[target_ind]
            

                
        # # Add the goal as the end of the route, then add to solution routes
        sub_route = [ distant_point ] + list(N[path]) + [ goal ]
        school_routes.append(sub_route)

        # Remove points in the path from the next iteration of routing
        points = np.delete(points, N_ind[path], 0)
        students = np.delete(students, N_ind[path], 0)
    return school_routes

def Tabu_Search(school_name, initial):
    data, goal = bus_router.sample(school_name)
    coordinates = list(data.keys())

    #Initialize the variables for the tabu search
    max_iter = 100
    tabu_list = {point: [0, 0] for point in coordinates}
    
    ideal_sol = initial
    ideal_score = sum(Route_Distance(route) for route in ideal_sol) + min(bus_router.test_routes(school_name, ideal_sol), 10)
    iter_sol = copy.deepcopy(ideal_sol)
    
    last_action = []
    iter_cnt = 0
    iter_scores = []
    while (not last_action is None) and (iter_cnt < max_iter):
        last_action = None
        score_diff = 5 # Set up a tolerance margin of 5 when considering possible solutions
        planned_action = []

        # We can assess the current iteration route data
        iter_dist = [Route_Distance(route) for route in iter_sol]
        iter_students = bus_router.test_routes(school_name, iter_sol)
        for a in range(len(coordinates)-1):
            # Tabu search without aspiration
            if(tabu_list[coordinates[a]][0] > iter_cnt):
                continue
            for b in range(len(coordinates)-(a+1)):
                if(tabu_list[coordinates[a+b+1]][0] > iter_cnt):
                    continue

                # Make a copy of the routes as well as the locations of the coordinates to swap
                change_sol = copy.deepcopy(iter_sol)
                pa = Find_Stop_Index(change_sol, coordinates[a])
                pb = Find_Stop_Index(change_sol, coordinates[a+b+1])

                # Swap the two positions
                temp = change_sol[pa[0]][pa[1]]
                change_sol[pa[0]][pa[1]] = change_sol[pb[0]][pb[1]]
                change_sol[pb[0]][pb[1]] = temp
                
                # Assess the new score, then compare it to the new one
                iter_score = (iter_dist[pa[0]] + iter_dist[pb[0]]) + min(iter_students, 10)
                
                change_dist = Route_Distance(change_sol[pa[0]]) + Route_Distance(change_sol[pb[0]])
                change_students = bus_router.test_routes(school_name, change_sol)
                change_score = change_dist + min(change_students, 10)

                # The best score is kept for future use
                result = change_score - iter_score
                if(result < score_diff):
                    score_diff = result
                    planned_action = [coordinates[a], coordinates[a+b+1]]

        if(len(planned_action) > 0):
            last_action = planned_action
            # Log the nodes in this action as tabu for 20 iterations
            tabu_list[planned_action[0]][0] = 20 + iter_cnt
            tabu_list[planned_action[1]][0] = 20 + iter_cnt
            tabu_list[planned_action[0]][1] += 1
            tabu_list[planned_action[1]][1] += 1

            # Get the positions of the coordinates, then perform the swap
            pa = Find_Stop_Index(iter_sol, planned_action[0])
            pb = Find_Stop_Index(iter_sol, planned_action[1])

            temp = iter_sol[pa[0]][pa[1]]
            iter_sol[pa[0]][pa[1]] = iter_sol[pb[0]][pb[1]]
            iter_sol[pb[0]][pb[1]] = temp

            iter_score = sum(Route_Distance(route) for route in iter_sol) + min(bus_router.test_routes(school_name, iter_sol), 10)
            iter_scores.append(iter_score)
            # Compare the current solution to the ideal one, replace if the current is better
            if(iter_score < ideal_score):
                ideal_sol = iter_sol
                ideal_score = iter_score
        print(".", end=" ")
        iter_cnt += 1

    # Plot the score progression
    #plt.plot(range(iter_cnt), iter_scores)
    #plt.show()
    
    print("!")
    return ideal_sol

def Annealing_Search(school_name):
    data, goal = bus_router.sample(school_name)
    coordinates = list(data.keys())
    
    # Randomly grab points to create a route
    sol = []
    sample_vals = np.array([data[key] for key in coordinates])
    sample_coords = copy.deepcopy(coordinates)
    while(sum(sample_vals) > 0):
        new_route = []
        space = bus_size

        # Add routes if there's space in the bus
        while(space > 0):
            # If no candidate is available, exit
            candidates = sample_vals - space
            if(len(candidates[candidates <= 0]) == 0):
                break

            # Select a value to take, and get an index to use
            selected = random.choice(sample_vals[candidates <= 0])
            index = np.where(sample_vals==selected)[0][0]

            # Remove the value            
            space -= selected
            sample_vals = np.delete(sample_vals, index)

            # Remove the coordinate to use
            new_route.append( sample_coords[index] )
            sample_coords = np.delete(sample_coords, index, 0)
        # Add the goal as the end of the route, then add to solution routes
        new_route.append(goal)
        sol.append(new_route)

    # Run annealing search for each route to properly align them
    total_iters = 0;
    annealing_scores = []
    for x in range(len(sol)):
        route = sol[x]
        # If a bus route only has one stop (total 2 stops)
        if(len(route) <= 2):
            continue
        
        # Initialize temperature variables
        temp = 1.0
        final_temp = 0.01
        decay_rate = 0.002
        score = Route_Distance(route)
        iters = 0;

        route_scores = []
        # Perform Annealing Search
        while(temp > final_temp):
            temp_route = copy.deepcopy(route)

            # Pick a point to remove
            target_pos = random.randint(0,len(temp_route)-2)
            selected_point = temp_route[target_pos]
            
            temp_route.pop(target_pos)

            # Reinsert randomly in an index
            new_pos = random.randint(0,len(temp_route)-2)
            temp_route.insert(new_pos, 0)
            temp_route[new_pos] = np.array(selected_point)
                     
            # Assess the new score, then compare it to the new one
            temp_score = Route_Distance(temp_route)
            annealing_scores.append(temp_score)
            
            # If the selected solution is worse, try the acceptance probability
            change = temp_score - score
            accept = (change < 0)
            if(change >= 0):
                seed = random.random()
                accept_prob = math.exp(-(change/(temp)))
                if(seed < accept_prob):
                    accept = True

            # For better solutions (or those that pass acceptance probability, update solution
            if(accept):
                route = temp_route
                score = temp_score

            iters += 1
            total_iters += 1
            if(iters % 10000 == 0):
                print(".", end=" ")
            
            # Decrease temperature according to decay rules
            temp = temp/(1+decay_rate*temp)

        sol[x] = route
        print("?", end=" ")

    # Plot score of routes
    #plt.plot(range(total_iters), annealing_scores)
    #plt.show()
    
    print("!")
    return sol

def Pheno_to_Geno(solution, reference, end):
    chromosome = []
    for route in solution:
        for point in route:
            index = -1
            if((point[0], point[1]) != end):
                index = reference.index((point[0], point[1]))
            chromosome.append(index)
    return chromosome
def Geno_to_Pheno(chromosome, reference, end):
    solution = []
    route = []
    for index in chromosome:
        if(index > -1):
            route.append(np.array(reference[index]))
        else:
            route.append(np.array(end))
            solution.append(route)
            route = []
    return solution
def Genetic_Algorithm(school_name, pop_size=20, generations=500, elite_num=2,
        prob_c=0.5, prob_m=0.1, crossovers=1, mutations=1):
    data, goal = bus_router.sample(school_name)
    points = list(data.keys())
    vals = [data[key] for key in points]

    population = []
    # Population
    for x in range(pop_size):        
        # Randomly grab points to create a route
        sol = []
        sample_coords = np.array(copy.deepcopy(points))
        sample_vals = np.array(copy.deepcopy(vals))
        while(sum(sample_vals) > 0):
            sub_route = []

            # Find most distant node
            distances = np.sqrt(np.square(goal[0]-sample_coords[:,0]) + np.square(goal[1]-sample_coords[:,1]))
            sorted_indices = np.argsort(distances)
            distant_point = sample_coords[sorted_indices[-1]]

            # Start a bus route from there
            space = bus_size - sample_vals[sorted_indices[-1]]
            sample_coords = np.delete(sample_coords, sorted_indices[-1], 0)
            sample_vals = np.delete(sample_vals, sorted_indices[-1], 0)

            # Generate a sublist of neighbour nodes
            #Distance between 'neighbour' to distant node does not exceed distance between distant & school
            max_N_dist = distances[sorted_indices[-1]]
            N_dist = np.sqrt(np.square(distant_point[0]-sample_coords[:,0]) + np.square(distant_point[1]-sample_coords[:,1]))

            # Remove distant nodes & nodes that have too many students
            # Get indices, the point, students on point, and distance from goal
            N_ind = np.where( (N_dist <= max_N_dist) & (sample_vals <= space) )[0]
            N = sample_coords[N_ind]
            N_stud = sample_vals[N_ind]
            goal_dist = np.sqrt(np.square(goal[0]-N[:,0]) + np.square(goal[1]-N[:,1]))

            curr = distant_point
            path = []
            # Add routes if there's space in the bus
            while(space > 0):
                # Compute heuristic potential
                dists = np.sqrt(np.square(curr[0]-N[:,0]) + np.square(curr[1]-N[:,1]))
                potential = dists/(1 + N_stud) + goal_dist/(1 + space)
                potential[N_stud > space] = 0
                potential[path] = 0 # Tabu: Do not revisit previous nodes

                # Addition: Grab 3 most potential nodes & pick from there
                sort_pot = np.argsort(potential)
                top_n = 5
                if(len(potential) > top_n):
                    potential = potential[sort_pot[-1-top_n:-1]]

                potential = potential[potential > 0]
                    
                if(sum(potential) <= 0):
                    #print("Route concluding early")
                    break

                probs = potential/sum(potential)
                    
                # Pick a random node to enter
                seed = random.random()
                for s in range(len(probs)):
                    seed -= probs[s]
                    if(seed <= 0):
                        break
                target_ind = sort_pot[-1-s]

                # Add target node to path & update position to node
                path.append(target_ind)
                curr = N[target_ind]
                space -= N_stud[target_ind]
            # # Add the goal as the end of the route, then add to solution routes
            # sub_route.append(np.array(goal))
            sub_route = [ distant_point ] + list(N[path]) + [ goal ]
            sol.append(sub_route)

            # Remove points in the path from the next iteration of routing
            sample_vals = np.delete(sample_vals, N_ind[path], 0)
            sample_coords = np.delete(sample_coords, N_ind[path], 0)
            
        #print(sum(Route_Distance(route) for route in sol))
        population.append(Pheno_to_Geno(sol,points,goal))

    best_fitnesses = []
    gen_fitnesses = []
    # Run Algorithm for a fixed number of generations
    for x in range(generations):
        if(x % 50 == 0):
            print(".", end=" ")
        
        # Fitness
        scores =  [ sum(Route_Distance(route) for route in Geno_to_Pheno(solution, points, goal)) for solution in population ]
        gen_fitnesses.append(sum(scores)/len(scores))
        fitnesses = [ 1/sum(Route_Distance(route) for route in Geno_to_Pheno(solution, points, goal)) for solution in population ]

        # Generate probability wheel
        total_fitness = sum(fitnesses)
        for i in range(pop_size):
            fitnesses[i] /= total_fitness
            if(i > 0):
                fitnesses[i] += fitnesses[i-1]

        # Selection
        next_gen = []
        # Guarantee elite individuals survive
        sorted_fitnesses = np.argsort(fitnesses)
        for i in range(elite_num):
            next_gen.append(population[sorted_fitnesses[-1-i]])
        # Record best score
        best_sol = population[sorted_fitnesses[-1]]
        best_fitnesses.append(sum([Route_Distance(route) for route in Geno_to_Pheno(best_sol, points, goal)]))
        
        # Random roll to get an index to use
        for i in range(pop_size-elite_num):
            seed = random.random()
            for j in range(pop_size):
                if(fitnesses[j] > seed):
                    j -= 1
                    if(j < 0):
                        j = 0
                    break
            next_gen.append(population[j])
        
        # Crossover
        for i in range(round(pop_size/2)):
            # Random roll to see if we can skip crossover
            seed = random.random()
            if(seed > prob_c):
                continue
            
            par_1 = next_gen[i*2]
            par_2 = next_gen[i*2 + 1]
            
            # Determine indices where route ends
            ends_1 = [itr+1 for itr,val in enumerate(par_1) if val<0]
            ends_2 = [itr+1 for itr,val in enumerate(par_2) if val<0]
            ends_1.insert(0,0)
            ends_2.insert(0,0)

            # For each child, add routes from one of the parents
            child_1 = par_2[ends_2[0]:ends_2[crossovers]]
            child_2 = par_1[ends_1[0]:ends_1[crossovers]]

            # Populate each child by maintaining the order from one parent
            space = bus_size
            for ind in par_1:
                if(ind in child_1):
                    continue
                # If bus cannot fit students at a stop, start a new route
                if(space < vals[ind]):
                    child_1.append(-1)
                    space = bus_size
                space -= vals[ind]
                child_1.append(ind)
            child_1.append(-1)

            space = bus_size
            for ind in par_2:
                if(ind in child_2):
                    continue
                # If bus cannot fit students at a stop, start a new route
                if(space < vals[ind]):
                    child_2.append(-1)
                    space = bus_size
                space -= vals[ind]
                child_2.append(ind)
            child_2.append(-1)
            
            next_gen[i*2] = child_1
            next_gen[i*2 + 1] = child_2

        # Mutation
        for i in range(pop_size):
            # Random roll to see if we can skip crossover
            seed = random.random()
            if(seed > prob_m):
                continue

            # Obtain info about the routes in the solution
            par = next_gen[i]
            ends = [itr+1 for itr,val in enumerate(par) if val<0]
            ends.insert(0,0)
            
            # Swap random pairs of points for each route
            for j in range(len(ends)-1):
                route = par[ends[j]:ends[j+1]-1]
                for m in range(mutations):
                    swap_vals = np.random.choice(route, 2)
                    ind_a = par.index(swap_vals[0])
                    ind_b = par.index(swap_vals[1])
                    par[ind_a] = swap_vals[1]
                    par[ind_b] = swap_vals[0]
                
            next_gen[i] = par

    # Present fitnesses over time
    #plt.plot(range(generations), gen_fitnesses)
    #plt.show()
    #plt.clf()
    #plt.plot(range(generations), best_fitnesses)
    #plt.show()
    print("!")
    # Find best child from final generation to return as answer
    solution_pop = [ Geno_to_Pheno(solution, points, goal) for solution in population ]
    fitnesses = [ sum(Route_Distance(route) for route in sol) for sol in solution_pop ]
    sorted_fitnesses = np.argsort(fitnesses)
    
    return solution_pop[sorted_fitnesses[0]]

def Ant_System_Algorithm(school_name, num_ants=40, num_iters=1000):
    data, goal = bus_router.sample(school_name)
    school_routes = []

    # Put the coordinates & students in each stop into an array
    points = np.array(list(data.keys()))
    students = np.fromiter(data.values(), dtype=int)

    while(np.sum(students) > 0):
        # Find most distant node to start a route from
        distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))
        sorted_indices = np.argsort(distances)
        distant_point = points[sorted_indices[-1]]

        # Remove distant node from list
        remaining_seats = bus_size - students[sorted_indices[-1]]
        points = np.delete(points, sorted_indices[-1], 0)
        students = np.delete(students, sorted_indices[-1], 0)

        # Generate a sublist of neighbour nodes
        #Distance between 'neighbour' to distant node does not exceed distance between distant & school
        max_N_dist = distances[sorted_indices[-1]]
        N_dist = np.sqrt(np.square(distant_point[0]-points[:,0]) + np.square(distant_point[1]-points[:,1]))

        # Remove distant nodes & nodes that have too many students
        N_ind = np.where( (N_dist <= max_N_dist) & (students <= remaining_seats) )[0]
        N = points[N_ind]
        N_stud = students[N_ind]
        
        goal_dist = np.sqrt(np.square(goal[0]-N[:,0]) + np.square(goal[1]-N[:,1]))
        
        # At this point, generate a (N+1)x(N) array for pheromones
        #N+1 at first dimension as last row represents start node
        base_value = 5.0
        rein_value = 5/num_ants
        pher = np.ones((len(N)+1, len(N)))*base_value
        for x in range(len(N)):
            pher[x, x] = 0

        for i in range(num_iters):
            # Simulate anthill's ants heading out
            for ant in range(num_ants):
                curr = distant_point
                c_ind = len(N)
                path = []
                open_seats = remaining_seats
                # Simulate an ant navigating from start to end
                while(open_seats > 0):
                    # Compute heuristic potential
                    dists = np.sqrt(np.square(curr[0]-N[:,0]) + np.square(curr[1]-N[:,1]))
                    potential = dists/(1 + N_stud) + goal_dist/(1 + open_seats)
                    potential[N_stud > open_seats] = 0
                    potential[path] = 0 # Tabu: Do not revisit previous nodes
                    if(sum(potential) <= 0):
                        #print("Route concluding early")
                        break

                    # Combine pheromone & heuristic to make probability
                    probs = pher[c_ind,:] * potential
                    probs /= sum(probs)
                    
                    # Pick a random node to enter
                    seed = random.random()
                    for s in range(len(probs)):
                        seed -= probs[s]
                        if(seed <= 0):
                            break

                    # Move to selected node
                    path.append(s)
                    curr = N[s]
                    open_seats -= N_stud[s]
                    c_ind = s
                # With complete path, compute route cost & pheromone to add back into syste,
                route = [ distant_point ] + list(N[path]) + [ goal ]
                award = rein_value / (Route_Distance(route))
                
                # Backtrack through route to update the values
                for u in range(len(path)):
                    x,y = (len(N), path[-1-u])
                    if(u < len(path)-1):
                        x = path[-2-u]
                    pher[x,y] += award
            # Decay function for pheromones
            #if(i%100 == 0):
            #    print(pher[70,:])
            pher *= 0.99
        #print(pher[70,:])

        '''
        if(len(N) < 25):
            # Visualize the pheromones in a scatterplot
            X = np.append(points[:,0], goal[0])
            Y = np.append(points[:,1], goal[1])
            S = np.append(students, 50)
            c = np.append([0 for x in range(len(X)-1)], 250)
            plt.clf()
            plt.scatter(X, Y, S, c=c)

            for a in range(len(pher)):
                pointA = distant_point
                if( a < len(pher)-1 ):
                    pointA = N[a]

                total_pher = sum(pher[a])
                for b in range(len(pher[a])):
                    pointB = N[b]

                    x_line = [pointA[0], pointB[0]]
                    y_line = [pointA[1], pointB[1]]
                    plt.plot(x_line, y_line, "ro-", linewidth=(pher[a][b]/total_pher)*2, markersize=0)

            plt.xlabel("Latitude (°)")
            plt.ylabel("Longitude (°)")
            plt.show()
        '''
        # Add route to school routes & remove related nodes from future queries
        school_routes.append(route)
        points = np.delete(points, N_ind[path], 0)
        students = np.delete(students, N_ind[path], 0)
        print(str(Route_Distance(route))+".", end=" ")
        

    print("!")
    
    return school_routes

def Ant_Tabu_Search(school_name):
    data, goal = bus_router.sample(school_name)
    school_routes = []

    # Put the coordinates & students in each stop into an array
    points = np.array(list(data.keys()))
    students = np.fromiter(data.values(), dtype=int)
    anthill_routes = []
    final_routes = []

    while(np.sum(students) > 0):
        # Find most distant node to start a route from
        distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))
        sorted_indices = np.argsort(distances)
        distant_point = points[sorted_indices[-1]]

        # Remove distant node from list
        remaining_seats = bus_size - students[sorted_indices[-1]]
        points = np.delete(points, sorted_indices[-1], 0)
        students = np.delete(students, sorted_indices[-1], 0)

        # Generate a sublist of neighbour nodes
        #Distance between 'neighbour' to distant node does not exceed distance between distant & school
        max_N_dist = distances[sorted_indices[-1]]
        N_dist = np.sqrt(np.square(distant_point[0]-points[:,0]) + np.square(distant_point[1]-points[:,1]))

        # Remove distant nodes & nodes that have too many students
        N_ind = np.where( (N_dist <= max_N_dist) & (students <= remaining_seats) )[0]
        N = points[N_ind]
        N_stud = students[N_ind]
        
        goal_dist = np.sqrt(np.square(goal[0]-N[:,0]) + np.square(goal[1]-N[:,1]))

        # Adaptive behavior, set iterations & anthill ants based on neighbourhood size
        anthill_iters = max( round(pow(len(N), 13/8)), 500 )
        anthill_ants = max( round(len(N)/2), 20 )
        
        # At this point, generate a (N+1)x(N) array for pheromones
        #N+1 at first dimension as last row represents start node
        base_value = 5.0
        rein_value = 5/anthill_ants
        pher = np.ones((len(N)+1, len(N)))*base_value
        for x in range(len(N)):
            pher[x, x] = 0

        for i in range(anthill_iters):
            # Simulate anthill's ants heading out
            for ant in range(anthill_ants):
                curr = distant_point
                c_ind = len(N)
                path = []
                open_seats = remaining_seats
                # Simulate an ant navigating from start to end
                while(open_seats > 0):
                    # Compute heuristic potential
                    dists = np.sqrt(np.square(curr[0]-N[:,0]) + np.square(curr[1]-N[:,1]))
                    potential = dists/(1 + N_stud) + goal_dist/(1 + open_seats)
                    potential[N_stud > open_seats] = 0
                    potential[path] = 0 # Tabu: Do not revisit previous nodes
                    if(sum(potential) <= 0):
                        #print("Route concluding early")
                        break

                    # Combine pheromone & heuristic to make probability
                    #Probability gets weighting towards pheromones at end of algorithm
                    probs = pow(pher[c_ind,:], 1+i/anthill_iters) * potential
                    probs /= sum(probs)
                    
                    # Pick a random node to enter
                    seed = random.random()
                    for s in range(len(probs)):
                        seed -= probs[s]
                        if(seed <= 0):
                            break

                    # Move to selected node
                    path.append(s)
                    curr = N[s]
                    open_seats -= N_stud[s]
                    c_ind = s
                # With complete path, compute route cost & pheromone to add back into syste,
                route = [ distant_point ] + list(N[path]) + [ goal ]
                award = rein_value / (Route_Distance(route))
                
                # Backtrack through route to update the values
                for u in range(len(path)):
                    x,y = (len(N), path[-1-u])
                    if(u < len(path)-1):
                        x = path[-2-u]
                    pher[x,y] += award
            # Decay function for pheromones
            if(i%100 == 0):
                print(">", end=" ")
            pher *= 0.99
        
        anthill_routes.append(Route_Distance(route))
        # Apply tabu search on route
        tabu_iters = pow( len(route), 3/2 )
        tabu_time = round( len(route) * 3/8 )

        tabu_list = [[0, 0] for y in range(len(route)-1)]
        
        tabu_sol = copy.deepcopy(route)
        tabu_score = Route_Distance(route)
        tabit_sol = copy.deepcopy(tabu_sol)
        
        last_action = []
        tab_iter = 0
        tabu_scores = []
        while (not last_action is None) and (tab_iter < tabu_iters):
            last_action = None
            score_diff = 1 # Set up a tolerance margin of 5 when considering possible solutions
            planned_action = []

            tabit_score = Route_Distance(tabit_sol)
            # We can assess the current iteration route data
            for a in range(len(route)-3):
                # Tabu search without aspiration
                if(tabu_list[a][0] > tab_iter):
                    continue
                for b in range(a+1, len(route)-2):
                    if(tabu_list[b][0] > tab_iter):
                        continue

                    # Swap the two points in a route
                    tabcha_sol = copy.deepcopy(tabit_sol)
                    
                    temp = tabcha_sol[a]
                    tabcha_sol[a] = tabcha_sol[b]
                    tabcha_sol[b] = temp
                    
                    # Assess the new score, then compare it to the new one
                    tabcha_score = Route_Distance(tabcha_sol)
                    result = tabcha_score - tabit_score
                    
                    # The best score is kept for future use
                    
                    if(result < score_diff):
                        score_diff = result
                        planned_action = [a, b]

            if(len(planned_action) > 0):
                last_action = planned_action
                a, b = planned_action
                # Log the nodes in this action as tabu for 20 iterations
                tabu_list[a][0] = tabu_time + tab_iter
                tabu_list[b][0] = tabu_time + tab_iter
                tabu_list[a][1] += 1
                tabu_list[b][1] += 1

                # Get the positions of the coordinates, then perform the swap
                temp = tabit_sol[a]
                tabit_sol[a] = tabit_sol[b]
                tabit_sol[b] = temp

                tabit_score = Route_Distance(tabit_sol)
                # Compare the current solution to the ideal one, replace if the current is better
                if(tabit_score < tabu_score):
                    tabu_sol = tabit_sol
                    tabu_score = tabit_score

                
                tabu_scores.append(tabu_score)
            if(tab_iter % 100 == 0):
                print(".", end=" ")
            tab_iter += 1

        # Plot the score progression
        #plt.plot(range(len(tabu_scores)), tabu_scores)
        #plt.xlabel("Iterations")
        #plt.ylabel("Route Distance (°)")
        #plt.show()
        final_routes.append(Route_Distance(tabu_sol))
        
        # Add route to school routes & remove related nodes from future queries
        school_routes.append(tabu_sol)
        points = np.delete(points, N_ind[path], 0)
        students = np.delete(students, N_ind[path], 0)
        print("?", end=" ")
        #print(bus_router.test_routes(school_name, school_routes))

    # Gauge performance of route
    '''
    ind = np.arange(len(anthill_routes))
    width = 0.35
    ants = plt.bar(ind, anthill_routes, width, label="Anthill Route Distance")
    finals = plt.bar(ind+width, final_routes, width, label="Finalized Route Distance")
    plt.legend(handles=[ants, finals])
    plt.xlabel("Route")
    plt.ylabel("Distance (°)")
    plt.show()
    '''
    print("!")
    
    return school_routes

#for x in range(5):
#    Best_First_Search(x)
#Best_First_Search(0, True)
start = time.perf_counter()

'''
for x in range(10):
    school_name = school_keys[x]
    data, goal = bus_router.sample(school_name)

    # Put the coordinates & students in each stop into an array
    points = np.array(list(data.keys()))
    students = np.fromiter(data.values(), dtype=int)

    distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))

    print(len(students), sum(students), max(students),
          min(students), int(round( sum(students)/len(students) )),
          max(distances), min(distances), sum(distances)/len(distances) )
'''

times = []
scores = []
'''
print("Graph Search")
for x in range(10):
    school_name = school_keys[x]
    start = time.perf_counter()
    sol = Best_First_Search(school_name)
    end = time.perf_counter()
    times.append(end-start)
    scores.append(sum(Route_Distance(route) for route in sol))
    print("Search complete for School #" + str(x+1) + ". Time elapsed: "+str(end-start) + "; Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, sol)))
'''

times.append(0)
scores.append(0)
'''
print("Tabu Search")
for x in range(1, 10):
    school_name = school_keys[x]
    init = Best_First_Search(school_name)
    start = time.perf_counter()
    sol = Tabu_Search(school_name, init)
    end = time.perf_counter()
    times.append(end-start)
    scores.append(sum(Route_Distance(route) for route in sol))
    print("Search complete for School #" + str(x+1) + ". Time elapsed: "+str(end-start) + "; Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, sol)))
'''
'''
print("Annealing Search")
for x in range(1, 10):
    school_name = school_keys[x]
    start = time.perf_counter()
    sol = Annealing_Search(school_name)
    end = time.perf_counter()
    times.append(end-start)
    scores.append(sum(Route_Distance(route) for route in sol))
    print("Search complete for School #" + str(x+1) + ". Time elapsed: "+str(end-start) + "; Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, sol)))
'''
'''
print("Genetic Search")
for x in range(1, 10):
    school_name = school_keys[x]
    start = time.perf_counter()
    sol = Genetic_Algorithm(school_name)
    end = time.perf_counter()
    times.append(end-start)
    scores.append(sum(Route_Distance(route) for route in sol))
    print("Search complete for School #" + str(x+1) + ". Time elapsed: "+str(end-start) + "; Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, sol)))
'''
'''
print("Anthill Search")
for x in range(1, 10):
    school_name = school_keys[x]
    start = time.perf_counter()
    sol = Ant_System_Algorithm(school_name)
    end = time.perf_counter()
    times.append(end-start)
    scores.append(sum(Route_Distance(route) for route in sol))
    print("Search complete for School #" + str(x+1) + ". Time elapsed: "+str(end-start) + "; Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, sol)))
'''
print("Anthill (with Tabu Search) Search")
for x in range(1, 10):
    school_name = school_keys[x]
    start = time.perf_counter()
    sol = Ant_Tabu_Search(school_name)
    end = time.perf_counter()
    times.append(end-start)
    scores.append(sum(Route_Distance(route) for route in sol))
    print("Search complete for School #" + str(x+1) + ". Time elapsed: "+str(end-start) + "; Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, sol)))


for x in range(len(times)):
    print(x+1, times[x], scores[x])    

'''
school_name = school_keys[0]
init_sol = Best_First_Search(school_name)
bfs_time = time.perf_counter()
print("Best First Search Completed. Time elapsed: "+str(bfs_time-start))
print("Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in init_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, init_sol)))
initial = copy.deepcopy(init_sol)
#Draw_Bus_Routes(school_name, init_sol)

tabu_sol = Tabu_Search(school_name, initial)
tabu_time = time.perf_counter()
print("Tabu Search Completed. Time elapsed: "+str(tabu_time-bfs_time))
print("Tabu Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in tabu_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, tabu_sol)))
Draw_Bus_Routes(school_name, tabu_sol)

anneal_sol = Annealing_Search(school_name)
anneal_time = time.perf_counter()
print("Simulated Annealing Search Completed. Time elapsed: "+str(anneal_time-bfs_time))
print("Simulated Annealing Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in anneal_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, anneal_sol)))
#Draw_Bus_Routes(school_name, anneal_sol)

gene_sol = Genetic_Algorithm(school_name, pop_size=20, generations=500, elite_num=2,
        prob_c=0.5, prob_m=0.1, crossovers=1, mutations=1)
gene_time = time.perf_counter()
print("Genetic Algorithm Search Completed. Time elapsed: "+str(gene_time-bfs_time))
print("Genetic Algorithm Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in gene_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, gene_sol)))
Draw_Bus_Routes(school_name, gene_sol)

aco_sol = Ant_System_Algorithm(school_name, num_ants=40, num_iters=1000)
aco_time = time.perf_counter()
print("Ant System Algorithm Search Completed. Time elapsed: "+str(aco_time-bfs_time))
print("Ant System Algorithm Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in aco_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, aco_sol)))
Draw_Bus_Routes(school_name, aco_sol)

ats_sol = Ant_Tabu_Search(school_name)
ats_time = time.perf_counter()
print("Ant System Algorithm Search Completed. Time elapsed: "+str(ats_time-bfs_time))
print("Ant System Algorithm Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in ats_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, ats_sol)))
Draw_Bus_Routes(school_name, ats_sol)
'''

"""Test Case: reading every coordinate provided
for item in coordinates:
    print(item)
    print(item[0])
    print(item[1])
    print(bus_router.get(item))
"""
"""Test Case: grab specific coordinates
print(bus_router.get([174.82397, -36.90762667]))
print(bus_router.get([174.736683, -36.737197]))
print(bus_router.get([1, 1]))
"""
