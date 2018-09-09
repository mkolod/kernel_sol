#!/bin/env python3



import cxxfilt, sqlite3, os, sys, subprocess

# TODO: make this a context managet so user doesn't need to remember to close cursor and connection

class DB(object):

	def __init__(self, dbFile):

		assert os.path.exists(dbFile)

		try:

			conn = sqlite3.connect(dbFile)

			c = conn.cursor()

		except:

			print("Error opening {}".format(dbFile))

			sys.exit(1)



		self.conn = conn

		self.c = c



	def select(self, cmd):

		try:

			self.c.execute(cmd)

			rows = self.c.fetchall()

		except:

			print("Uncaught error in SQLite access while executing {}".format(cmd))

			sys.exit(1)



		#print(rows)

		return rows



	def close(self):

		self.c.close()

		self.conn.close()



#def demangle(name):
#
#	cmd = "c++filt {}".format(name)
#
#	result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,shell=True).decode('ascii').strip()
#
#	return result



db = DB(sys.argv[1])

kernels = []

kernelNames = []

corrId = []

startTime = []

marker = []

markerNames = []



#Get kernels

cmd = "select name,correlationId from CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;" # limit 10"



result = db.select(cmd)
print(result)


for k,id in result:

	kernels.append(k)

	corrId.append(id)



#Get correlation id start time

for id in corrId:

	cmd = "select start from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId={}".format(id);

	result = db.select(cmd)

	assert len(result) == 1

	startTime.append(result[0][0])



#Get the marker which started just before the starttime

for ts in startTime:

	cmd = "select name from CUPTI_ACTIVITY_KIND_MARKER as t1 where t1.timestamp < {} order by timestamp desc limit 1".format(ts)

	result = db.select(cmd)

	assert len(result) == 1

	print(result)

	marker.append(result[0][0])



for i in range(len(kernels)):

	#Get kernel name and marker name

	cmd = "select value from StringTable where _id_ = {}".format(kernels[i])

	r1 = db.select(cmd)

	assert len(r1) == 1

	kernelNames.append(cxxfilt.demangle(r1[0][0]))



	cmd = "select value from StringTable where _id_ = {}".format(marker[i])

	r1 = db.select(cmd)

	assert len(r1) == 1

	markerNames.append(r1[0][0])



	print("Kernel: {}".format(kernelNames[i]))

	print("Marker: {}".format(markerNames[i]))

	print()



db.close()
