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
    plt.show()
    

def Best_First_Search(school_name, cutoff = 0):
    data, goal = bus_router.sample(school_name)
    school_routes = []

    # Put the coordinates & students in each stop into an array
    points = np.array(list(data.keys()))
    students = np.fromiter(data.values(), dtype=int)
    if(cutoff > 0):
        points = points[:cutoff]
        students = students[:cutoff]

    routes = 0
    while(np.sum(students) > 0):
        route = []
        # Find most distant node to start a route from
        distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))
        sorted_indices = np.argsort(distances)
        distant_point = points[sorted_indices[-1]]
        route.append(distant_point)
        
        seats_available = bus_size
        if(students[sorted_indices[-1]] < seats_available):
            seats_available -= students[sorted_indices[-1]]
            students[sorted_indices[-1]] = 0
        else:
            students[sorted_indices[-1]] -= seats_available
            seats_available = 0 

        if(students[sorted_indices[-1]] == 0):
            points = np.delete(points, sorted_indices[-1], 0)
            students = np.delete(students, sorted_indices[-1], 0)
        
        # Navigate a route until we reach the goal
        target_node = distant_point
        while(seats_available > 0 and np.sum(students) > 0):#(target_node[0] != goal[0] and target_node[1] != goal[1]):
            # Use distance between current & target node, and target & goal node
            travel_distances = np.sqrt(np.square(distant_point[0]-points[:,0]) + np.square(distant_point[1]-points[:,1]))
            goal_distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))

            # Potential for next node:
            #  travel distance scores lower the more students there are in the stop (capped by bus space)
            #  goal distance scores lower the 
            potential = travel_distances/(1 + np.minimum(seats_available, students)) + goal_distances/(1 + seats_available)
            sorted_indices = np.argsort(potential)
            target_node = points[sorted_indices[0]]
            if(students[sorted_indices[0]] < seats_available):
                seats_available -= students[sorted_indices[0]]
                students[sorted_indices[0]] = 0
            else:
                students[sorted_indices[0]] -= seats_available
                seats_available = 0

            if(students[sorted_indices[0]] == 0):
                points = np.delete(points, sorted_indices[0], 0)
                students = np.delete(students, sorted_indices[0], 0)

            route.append(target_node)
            distant_point = target_node

        #print("Completed route: "+new_route_name)
        route.append(goal)
        school_routes.append(route)
        routes += 1

    return school_routes

def Tabu_Search(school_name, initial, cutoff=0):
    data, goal = bus_router.sample(school_name)
    coordinates = list(data.keys())
    if(cutoff > 0):
        coordinates = coordinates[:cutoff]

    #Initialize the variables for the tabu search
    max_iter = 100
    tabu_list = {point: [0, 0] for point in coordinates}
    
    ideal_sol = initial
    ideal_score = sum(Route_Distance(route) for route in ideal_sol) + min(bus_router.test_routes(school_name, ideal_sol), 10)
    iter_sol = copy.deepcopy(ideal_sol)
    
    last_action = []
    iter_cnt = 0
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
            # Compare the current solution to the ideal one, replace if the current is better
            if(iter_score < ideal_score):
                ideal_sol = iter_sol
                ideal_score = iter_score
        print(".", end=" ")
        iter_cnt += 1
            
    print("!")
    return ideal_sol

def Annealing_Search(school_name, cutoff=0):
    data, goal = bus_router.sample(school_name)
    coordinates = list(data.keys())
    if(cutoff > 0):
        coordinates = coordinates[:cutoff]
    
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
            
            # If the selected solution is worse, try the acceptance probability
            change = temp_score - score
            if(change >= 0):
                seed = random.random()
                accept_prob = math.exp(-(change/(temp)))
                if(seed >= accept_prob):
                    continue

            # For better solutions (or those that pass acceptance probability, update solution
            route = temp_route
            score = temp_score

            # Decrease temperature according to decay rules
            temp = temp/(1+decay_rate*temp)
            iters += 1
            if(iters % 10000 == 0):
                print(".", end=" ")
        sol[x] = route
        print("?", end=" ")

    print("!")
    return sol

#for x in range(len(school_keys)):
#    Best_First_Search(x)
#Best_First_Search(0, True)
start = time.perf_counter()

school_name = school_keys[0]
init_sol = Best_First_Search(school_name)
bfs_time = time.perf_counter()
print("Best First Search Completed. Time elapsed: "+str(bfs_time-start))
print("Initial Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in init_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, init_sol)))
initial = copy.deepcopy(init_sol)

'''
#init_sol = Best_First_Search(school_name, cutoff=40)
#tabu_sol = Tabu_Search(school_name, init_sol, cutoff=40)

tabu_sol = Tabu_Search(school_name, initial)
tabu_time = time.perf_counter()
print("Tabu Search Completed. Time elapsed: "+str(tabu_time-bfs_time))
print("Tabu Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in tabu_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, tabu_sol)))
Draw_Bus_Routes(school_name, tabu_sol)
'''

anneal_sol = Annealing_Search(school_name)
anneal_time = time.perf_counter()
print("Simulated Annealing Search Completed. Time elapsed: "+str(anneal_time-bfs_time))
print("Simulated Annealing Statistics - Total Distance: "+str(sum(Route_Distance(route) for route in anneal_sol))+" ; Missed Students: "+str(bus_router.test_routes(school_name, anneal_sol)))
Draw_Bus_Routes(school_name, anneal_sol)

  
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
