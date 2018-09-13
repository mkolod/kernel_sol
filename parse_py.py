#!/bin/env python3



import sqlite3, os, sys, subprocess, re, struct, binascii

from termcolor import colored



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

		except sqlite3.Error as e:

			print(e)

			sys.exit(1)

		except:

			print("Uncaught error in SQLite access while executing {}".format(cmd))

			sys.exit(1)



		#print(rows)

		return rows



	def close(self):

		self.c.close()

		self.conn.close()



def decode_object_id(obj_kind, obj_byte_array):

    '''

    Read in the object byte array

    The format is ProcID:ThreadID

    ProcID is 32 bits and threadID is 64 Bits

    The bytes in byte array are in reverse order

    '''

    pid   = 0

    th_id = 0

    if obj_kind != 2:

        print ("Error - unexpected obk_kind val -> {}, expecting 2".format(obj_kind))

        sys.exit(1)



    reverse_proc_id = obj_byte_array[:3]    ## Proc ID is 32 bits (4 bytes)

    reverse_th_id   = obj_byte_array[4:]    ## Thread ID is 64 bits - just take all the remaining bytes



    pid   = int.from_bytes(reverse_proc_id, byteorder='little')

    th_id = int.from_bytes(reverse_th_id,   byteorder='little')



    #print ("ProcId -> {} Thread ID -> {}".format(pid, th_id))



    return [pid, th_id]



def encode_object_id(pid, tid):

	#See the decode function above

	objId = struct.pack('<i', pid) + struct.pack('<q',tid)

	objId = binascii.hexlify(objId).decode('ascii').upper()

	return objId



def demangle(name):

	cmd = "c++filt {}".format(name)

	result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,shell=True).decode('ascii').strip()

	return result



def getString(id, demang=False):

	cmd = "select value from StringTable where _id_ = {}".format(id)

	result = db.select(cmd)

	assert len(result) == 1

	assert len(result[0]) == 1

	cadena = result[0][0]

	return demangle(cadena) if demang else cadena



def getMarkerInfo(objId, startTime, endTime):

	#Find all encapsulating markers

	#First find all "start marker" (flags=1) with timestamp before the start time

	cmd = 'SELECT id FROM CUPTI_ACTIVITY_KIND_MARKER WHERE HEX(objectId) = "{}" AND timestamp < {} AND flags = 2'.format(objId, startTime)

	result = db.select(cmd)

	assert (len(result) > 0)

	#Create

	ids = list(map(lambda x : x[0], result))

	ids = ",".join(str(x) for x in ids)

	#print(ids)



	#For these ids get "stop markers" (flags=4) with timetamp after the stop time

	#Get the top most marker

	cmd = 'SELECT id FROM CUPTI_ACTIVITY_KIND_MARKER WHERE id IN ({}) AND HEX(objectId) = "{}" AND timestamp > {} AND flags = 4 ORDER BY timestamp DESC LIMIT 1'.format(ids, objId, endTime)

	result = db.select(cmd)



	if (len(result) == 0):

		#There is no encapsulating marker

		return 0

	else:

		#return the nameId of that marker

		assert (len(result) == 1)

		assert (len(result[0]) == 1)

		id = result[0][0]

		cmd = 'SELECT name from CUPTI_ACTIVITY_KIND_MARKER WHERE id = {} AND flags = 2'.format(id)

		result = db.select(cmd)

		assert (len(result) == 1)

		assert (len(result[0]) == 1)

		nameId = result[0][0]

		assert nameId != 0

		return nameId



class Kernel(object):

	kernels = []



	def __init__(self):

		self.kNameId = None

		self.kName = None

		self.kStartTime = None	#GPU start time

		self.kEndTime = None	#GPU end time

		self.kDuration = None

		self.corrId = None

		self.rStartTime = None	#CPU start time

		self.rEndTime = None	#CPU end time

		self.rDuration = None

		self.mNameId = None

		self.mName = None

		self.tid = None

		self.pid = None



		Kernel.kernels.append(self)



if len(sys.argv) != 2:

	print("Usage {} <nvvp db file>".format(sys.argv[0]))

	sys.exit(1)



db = DB(sys.argv[1])



#Get kernels of interest (say only convs)

#Can get more info if required

cmd = "select name,correlationId,start,end from CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"

result = db.select(cmd)

for k,id,start,end in result:

	kernel = Kernel()

	kernel.kNameId = int(k)

	kernel.corrId = int(id)

	start = int(start)

	end = int(end)

	assert end > start

	kernel.kStartTime = start

	kernel.kEndTime = end

	kernel.kDuration = end - start



#Get info for each kernel

for kernel in Kernel.kernels:

	#Get kernel name

	kernel.kName = getString(kernel.kNameId, True)

	

	#Get CPU/runtime start,end,thread id,process id

	cmd = "select start,end,processId,threadId from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId={}".format(kernel.corrId);

	result = db.select(cmd)

	assert len(result) == 1

	assert len(result[0]) == 4

	start = int(result[0][0])

	end = int(result[0][1])

	pid = int(result[0][2])

	tid = int(result[0][3])

	assert end > start

	kernel.rStartTime = start

	kernel.rEndTime = end

	kernel.rDuration = end - start

	kernel.pid = pid

	kernel.tid = tid & 0xffffffff	#convert to unsigned



	objId = encode_object_id(kernel.pid, kernel.tid)



	kernel.mNameId = getMarkerInfo(objId, kernel.rStartTime, kernel.rEndTime)



	'''

	#Get the closest "start marker" which matches a pattern

	cmd = 'SELECT name FROM CUPTI_ACTIVITY_KIND_MARKER AS a INNER JOIN StringTable AS b ON a.name=b._id_ WHERE HEX(objectId) = "{}" AND a.timestamp < {} AND value LIKE "%padd%" ORDER BY timestamp DESC LIMIT 1'.format(objId, kernel.rStartTime)

	else:

		#cmd = 'select name from CUPTI_ACTIVITY_KIND_MARKER as t1 where CAST(HEX(t1.objectId) AS CHAR) = "{}" AND t1.timestamp < {} order by timestamp desc limit 1'.format(objId, kernel.rStartTime)

		cmd = 'select name from CUPTI_ACTIVITY_KIND_MARKER as t1 where HEX(t1.objectId) = "{}" AND t1.timestamp < {} order by timestamp desc limit 1'.format(objId, kernel.rStartTime)



	result = db.select(cmd)

	assert len(result) == 1

	assert len(result[0]) == 1

	#pid,tid = decode_object_id(result[0][1], result[0][2])

	#print("pid {}, tid {}".format(pid,tid))

	kernel.mNameId = int(result[0][0])

	'''



	#kernel.mNameId = 0

	#Get marker name

	kernel.mName = getString(kernel.mNameId)



	#print(kernel.__dict__)

	if sys.stdout.isatty():

		print(

			colored(kernel.kName, 'green'),

			colored(kernel.mName, 'red')

		)

	else:

		print(kernel.kName, kernel.mName)



#Print info

#for kernel in Kernel.kernels:

#	print(kernel.__dict__)



db.close()
