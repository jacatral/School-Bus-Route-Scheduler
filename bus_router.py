from source.bus_route_data_processor import SchoolBusRouteDataBuilder
from data.bus_router_model import SchoolBusRouter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

def Best_First_Search(index, display_graph=False):
    school_name = school_keys[index]
    data, goal = bus_router.sample(school_name)
    school_routes = []

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

    routes = 0
    route_names = []
    while(np.sum(students) > 0):
        new_route_name = school_name + "_" + str(routes)
        bus_router.new_route(new_route_name)
        route_names.append(new_route_name)
        
        # Find most distant node to start a route from
        distances = np.sqrt(np.square(goal[0]-points[:,0]) + np.square(goal[1]-points[:,1]))
        sorted_indices = np.argsort(distances)
        distant_point = points[sorted_indices[-1]]
        
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
            bus_router.assign_route(new_route_name,distant_point,target_node)

            x_line = [distant_point[0], target_node[0]]
            y_line = [distant_point[1], target_node[1]]
            plt.plot(x_line, y_line, "ro-", markersize=0)
            distant_point = target_node
        target_node = goal
        bus_router.assign_route(new_route_name,distant_point,target_node)

        x_line = [distant_point[0], target_node[0]]
        y_line = [distant_point[1], target_node[1]]
        plt.plot(x_line, y_line, "ro-", markersize=0)
        #print("Completed route: "+new_route_name)
        routes += 1
    school_routes.append(route_names)
    result = bus_router.test(school_name, route_names)
    print(school_name + " bus routes leave out " + str(result) + " students")
    total_score = 0
    for x in range(len(route_names)):
        weight = bus_router.compute_route_cost(route_names[x])
        #print(route_names[x] + " travels a total of " + str(weight) + " degrees")
        total_score += weight
    total_distance = round(total_score*111,2)
    print(str(routes) + " bus routes used to cover " + school_name + ". The routes travel a total of " + str(total_distance) + " km")
    # Display path
    if(display_graph):
        plt.show()

for x in range(len(school_keys)):
    Best_First_Search(x)
Best_First_Search(0, True)
  
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
"""Test Case: establishing a route
bus_router.new_route("test")
bus_router.check_route("test", "inst0")
print(bus_router.compute_route_cost("test"))

bus_router.assign_route("test",[174.74725, -36.71222889],[174.736683, -36.737197])
print(bus_router.compute_route_cost("test"))

bus_router.assign_route("test", [174.736683, -36.737197],[1.0,1.5])
print(bus_router.compute_route_cost("test"))
bus_router.remove_route_point("test", [1.0,1.5])
print(bus_router.compute_route_cost("test"))

bus_router.check_route("test", "inst0")
bus_router.print_routes()

bus_router.remove_route("test")
"""
"""Test Case: testing school bus routes
bus_router.new_route("test")
bus_router.assign_route("test",[174.74725, -36.71222889],[174.736683, -36.737197])

print(bus_router.test("inst0", []))
miss = bus_router.test("inst0", ["test", "one"])
print("inst0 still has " + str(miss) + " students without a bus")
"""
