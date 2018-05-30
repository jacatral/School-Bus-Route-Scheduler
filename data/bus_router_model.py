from pathlib import Path
import csv
import math

class SchoolBusRouter(object):
    def __init__(self, stopfile, schoolfile, bus_size):
        self.stopfile = stopfile
        self.schoolfile = schoolfile
        self.geographic_map = {}
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

    # Grab data from the geographic map
    def get(self, point):
        if not((point[0], point[1]) in self.geographic_map.keys()):
            print("Invalid coordinates")
            return
        value = self.geographic_map[(point[0], point[1])]
        return value

    # Grab a subset of the main data for a given school, as well as the school coordinates (goal node)
    def sample(self, school_name):
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

        return school_map, self.schools[school_name]

    # See how many students are left over for a school, if provided a list of bus routes
    def test_routes(self, school_name, route_list):
        if not(school_name in self.schools.keys()):
            print("Invalid school name")
            return

        # Map out the points of interest (stops, school) for a given school
        school_map, goal = self.sample(school_name)

        # Simulate a bus going through each route to pick up students
        for i in range(len(route_list)):
            route = route_list[i]
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
