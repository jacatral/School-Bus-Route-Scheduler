from source.bus_route_data_processor import SchoolBusRouteDataBuilder
from data.bus_router_model import SchoolBusRouter
from pathlib import Path

# Sample case is information provided by Auckland, New Zealand
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


bus_router = SchoolBusRouter(stop_file, school_file, 50)
#Get all coordinates & school names
coordinates, schools = bus_router.prepare()
#print(schools)

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

bus_router.new_route("test")
bus_router.assign_route("test",[174.74725, -36.71222889],[174.736683, -36.737197])
bus_router.check_route("test", "inst0")
print(bus_router.test("inst0", []))
print(bus_router.test("inst0", ["test"]))
