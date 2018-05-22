import csv
import math
import numpy as np
import random
from pathlib import Path

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

class SchoolBusRouteDataBuilder(object):
    def __init__(self, latitude):
        self.school_roster_min = 2000
        self.school_bus_usage = 0.55
        self.school_bus_range_km = 10
        # One degree of latitude is 111km
        # One degree of longitude at the equator is 111.321km
        self.lat_to_km = 111
        self.lon_to_km = math.cos(latitude*(math.pi/180))*111.321

    # Build datasets from the sample data provided
    #  stop_file requires a csv with Latitude(STOPLAT) & Longitude(STOPLON) columns
    #  school_file requires a csv with Latitude, Longitude, Name, and Total School Roll
    def buildSampleData(self, stop_file, school_file):
        if(not (Path(stop_file).is_file() and Path(school_file).is_file())):
            print("Bus stop file or School file is not provided")
            return False

        bus_stops = []
        bus_stop_headers = ["Longitude", "Latitude"]

        with open(stop_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                bus_stop = [float(row["STOPLON"]), float(row["STOPLAT"])]
                bus_stops.append(bus_stop)
        number_stops = len(bus_stops)
        bus_stops = np.array(bus_stops)
        print("Number of bus stops provided: " + str(number_stops))

        # Read through the school data to find schools that reach a minimum value
        # Then use targeted schools as sample data  & put students in nearby bus stops
        school_headers = ["Name", "Longitude", "Latitude"]
        schools = [school_headers]
        count = 0

        with open(school_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if(int(row["Total School Roll"]) >= self.school_roster_min):
                    school_lon = float(row["Longitude"])*self.lon_to_km
                    school_lat = float(row["Latitude"])*self.lat_to_km
                    school = ["inst"+str(count), row["Longitude"], row["Latitude"]]
                    schools.append(school)
                    bus_stop_headers.append("inst"+str(count))
                    count += 1

                    distances = np.sqrt(np.square(school_lon - bus_stops[:,0]*self.lon_to_km) + np.square(school_lat - bus_stops[:,1]*self.lat_to_km))
                    sorted_dists = np.argsort(distances)
                    selected_indices = []
                    target_range = self.school_bus_range_km*1.0
                    for index in sorted_dists:
                        if(len(selected_indices) == 0 and distances[index] >= target_range):
                            target_range = round(distances[index]) + 5
                        if(distances[index] > target_range*1.0):
                            break
                        selected_indices.append(index)
            
                    bus_roster = round(int(row["Total School Roll"])*self.school_bus_usage)
                    stop_students = np.zeros(number_stops)
                    number_assignments = len(selected_indices)
                    stop_assigments = constrained_sum_sample_pos(number_assignments, bus_roster)
                    for x in range(number_assignments):
                        stop_students[selected_indices[x]] = stop_assigments[x]
                    bus_stops = np.c_[ bus_stops, stop_students ]
            bool_filter = np.array([np.sum(row[2:]) > 0 for row in bus_stops])
            bus_stops = bus_stops[bool_filter]

        print("Number of schools provided: " + str(count))
        print("Total number of bus stops utilized: "+str(bus_stops.shape[0]))

        bus_stops = np.vstack([np.array(bus_stop_headers), bus_stops])

        schoolFile = open("data/school-data.csv", "w")
        with schoolFile:
            writer = csv.writer(schoolFile)
            writer.writerows(schools)

        stopFile = open("data/stop-data.csv", "w")
        with stopFile:
            writer = csv.writer(stopFile)
            writer.writerows(np.asarray(bus_stops))

        print("Dataset 'school-data.csv' & 'stop-data.csv' created")
        return True
