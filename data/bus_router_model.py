from pathlib import Path
import csv
import math

class SchoolBusRouter(object):
    def __init__(self, stopfile, schoolfile, bus_size):
        self.stopfile = stopfile
        self.schoolfile = schoolfile
        self.geographic_map = {}
        self.bus_routes = {}
        self.schools = {}
        self.bus_capacity = bus_size

    def prepare(self):
        if(not (Path(self.stopfile).is_file() and Path(self.schoolfile).is_file())):
            print("Provided datasets is invalid")
            return
        # Populate a map with coordinates of the bus stops & schools
        self.geographic_map = {};
        with open(self.stopfile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                coordinates = [0.0, 0.0]
                students = {}
                for column in row:
                    if(column == "Longitude"):
                        coordinates[0] = float(row[column])
                    elif(column == "Latitude"):
                        coordinates[1] = float(row[column])
                    else:
                        students[column] = int(float(row[column]))
                self.geographic_map[coordinates[0], coordinates[1]] = students
        self.schools = {}
        with open(self.schoolfile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                coordinates = [0.0, 0.0]
                schoolname = ""
                for column in row:
                    if(column == "Longitude"):
                        coordinates[0] = float(row[column])
                    elif(column == "Latitude"):
                        coordinates[1] = float(row[column])
                    else:
                        schoolname = row[column]
                self.schools[schoolname] = (coordinates[0], coordinates[1])
                self.geographic_map[coordinates[0], coordinates[1]] = schoolname
        print("Stops & Schools loaded into a coordinate dictionary")
        return self.geographic_map.keys(), self.schools

    def new_route(self, route_name):
        self.bus_routes[route_name] = []

    def remove_route(self, route_name):
        self.bus_routes.pop(route_name, None)

    def print_routes(self):
        for route_name in self.bus_routes.keys():
            print(route_name + " : " + str(self.bus_routes[route_name]))

    # If the first point is in the route, append the coordinates of pointB after pointA
    #  if the second point is in, prepend the coordinates
    #  otherwise simply insert pointA, then pointB
    def assign_route(self, route_name, pointA, pointB):
        if(not route_name in self.bus_routes):
            print("Invalid route provided")
            return
        list_index = 0
        if(pointA in self.bus_routes[route_name]):
            list_index = self.bus_routes[route_name].index(pointA) + 1
            self.bus_routes[route_name].insert(list_index, pointB)
        elif(pointB in self.bus_routes[route_name]):
            list_index = self.bus_routes[route_name].index(pointB)
            self.bus_routes[route_name].insert(list_index, pointA)
        else:
            self.bus_routes[route_name].append(pointA)
            self.bus_routes[route_name].append(pointB)

    # Removal in routes is simply removing a bus stop in a provided route
    def remove_route_point(self, route_name, point):
        if(not route_name in self.bus_routes):
            print("Invalid route provided")
            return
        if(not point in self.bus_routes[route_name]):
            print("Coordinates not in the provided bus route")
        else:
            self.bus_routes[route_name].remove(point)

    # Return total distance traveled by the bus
    def compute_route_cost(self, route_name):
        if(not route_name in self.bus_routes):
            print("Invalid route provided")
            return
        total = 0
        for x in range(len(self.bus_routes[route_name])-1):
            pointA = self.bus_routes[route_name][x-1]
            pointB = self.bus_routes[route_name][x]
            print(pointA, pointB)
            distance = math.sqrt( (pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2 )
            total += distance
        return total

    # Grab data from the geographic map
    def get(self, point):
        if not((point[0], point[1]) in self.geographic_map.keys()):
            print("Invalid coordinates")
            return
        value = self.geographic_map[(point[0], point[1])]
        return value

    # Read into the routing properties of a bus route
    def check_route(self, route_name, school_name):
        if(not route_name in self.bus_routes):
            print("Invalid route provided")
            return
        if not(school_name in self.schools.keys()):
            print("Invalid school name")
            return
        
        route = self.bus_routes[route_name]
        if(len(route) > 0 and (route[-1][0] != self.schools[school_name][0] or route[-1][1] != self.schools[school_name][1])):
            print("Warning: Route " + route_name + " does not end at the school " + school_name)
        capacity = self.bus_capacity
        for x in range(len(route)-1):
            point = route[x]
            entry = self.get((point[0], point[1]))
            students = int(float(entry[school_name]))
            if(students <= capacity):
                capacity -= students
            else:
                capacity = 0
        print("Route " + route_name + " concludes its route with " + str(capacity) + " empty seats")

    # See how many students are left over for a school given its bus routes
    def test(self, school_name, school_routes):
        if not(school_name in self.schools.keys()):
            print("Invalid school name")
            return

        # Map out the points of interest (stops, school) for a given school
        school_map = {}
        for coordinate in self.geographic_map.keys():
            entry = self.get(coordinate)
            if (type(entry) is dict):
                count = entry[school_name]
                if(float(count) > 0.0):
                    school_map[coordinate] = int(float(count))

        # Simulate a bus going through each route to pick up students
        for i in range(len(school_routes)):
            route_name = school_routes[i]
            if(not route_name in self.bus_routes):
                print("Invalid route provided")
                continue
            route = self.bus_routes[route_name]
            if(route[-1][0] != self.schools[school_name][0] or route[-1][1] != self.schools[school_name][1]):
                print("Route " + route_name + " does not end at the school")
                continue
            capacity = self.bus_capacity
            for x in range(len(route)-1):
                point = route[x]
                if(school_map[point[0], point[1]] <= capacity):
                    capacity -= school_map[point[0], point[1]]
                    school_map[point[0], point[1]] = 0
                else:
                    school_map[point[0], point[1]] -= capacity
                    capacity = 0

        # Check if any students are still out of school
        absentees = 0
        for coordinate in school_map.keys():
            absentees += school_map[coordinate]
        return absentees
