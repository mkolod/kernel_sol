#!/bin/env python3



import cxxfilt, sqlite3, os, sys, subprocess, re, struct, binascii

from termcolor import colored
import operator

# N = batch size
# C = number of input channels
# H = height
# W = width
# K = number of output channels
# R = filter height
# S = filter width
# PP_H = padding for height (symmetric padding assumed)
# PP_W = padding for width (symmetric padding assumed)
# SS_H = stride for height (symmetric stride assumed)
# SS_W = stride for width (symmetric stride assumed)
def conv1d_2d_flops(N, C, H, W, K, R, S, PP_H, PP_W, SS_H, SS_W,
                    gpu_clock_mhz=1082,
                    num_sms=80, tensor_cores_per_sm=8, fmas_per_tensor_core=64, 
                    glob_mem_bandwidth_gb=900):
    
    # n multiplications and n-1 additions
    # fused multiply-add
    flops_per_instance = C*S - 1 if H is None else C*R*S - 1
    
    if H is None:
        # 1D conv
        num_instances_per_filter = ((W - S + 2 * PP_W) // SS_W + 1)
    else:
        num_instances_per_filter = (H - R + 2 * PP_H) // SS_H + 1
        num_instances_per_filter *= ((W - S + 2 * PP_W) // SS_W + 1)
                  
    flops_per_filter = num_instances_per_filter * flops_per_instance
    # mult. by number of filters
    total_flops_per_layer = flops_per_filter * K
    # mult. by batch size
    total_flops_per_batch = N * total_flops_per_layer
    
    max_fmas_per_sec =  num_sms * tensor_cores_per_sm * fmas_per_tensor_core * gpu_clock_mhz * 1e6
    
    gb = 1 << 30
    # fp16, so 2 bytes per element
    input_dims = N*C*W if H is None else N*C*H*W
    dram_time_input = input_dims / (glob_mem_bandwidth_gb * gb)
    dram_time_output = num_instances_per_filter * N * K / (glob_mem_bandwidth_gb * gb)
    total_dram_time_sec = dram_time_input + dram_time_output
    kernel_compute_time_seconds = total_flops_per_batch / max_fmas_per_sec
    # accounting for bandwidth- and compute-bound problems
    # memory requests are overlapped in compute_bound kernels
    total_kernel_time_seconds = max(kernel_compute_time_seconds, total_dram_time_sec)

    return total_flops_per_batch, total_kernel_time_seconds

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


def list_hotspots():
    query = """
        WITH query1 as (
          SELECT
            kernels.name as name_idx, 
            1.0*(kernels.end - kernels.start)/1e6 as duration_ms,
            kernel_names.value as kernel_name
          FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL as kernels
	  JOIN StringTable as kernel_names on kernel_names._id_ = kernels.name
       )
       SELECT query1.kernel_name as kernel_name, 
       SUM(query1.duration_ms) AS total_duration
       FROM query1
       GROUP BY(kernel_name) ORDER BY total_duration DESC;
    """
    result = db.select(query)
    return result

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
    return cxxfilt.demangle(name)

#	cmd = "c++filt {}".format(name)
#
#	result = subprocess.check_output(cmd, stderr=subprocess.STDOUT,shell=True).decode('ascii').strip()
#
#	return result



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

hs_names, hs_times = zip(*list_hotspots()) 
total_time = sum(hs_times)
percentages = [100.0 * time / total_time for time in hs_times]

#Get info for each kernel

kernel_dict = {}

for idx, hs_name in enumerate(hs_names):

	kernel_dict[demangle(hs_name)] = {
		'hotspot_num': idx,
		'total_time': hs_times[idx],
		'percent_of_total': percentages[idx],
		'instances': []
	}
#	print("Hotspot #{:3d}".format(idx))
#	print("##############")
#	print("total time: {:.6f} ms, pct of total time: {:0.3f}%".format(hs_times[idx], percentages[idx]))
#	print("Name: {}\n".format(demangle(hs_name)))

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

	kernel_dict[kernel.kName]['instances'].append(kernel)

	#print(kernel.__dict__)

sorted_k_dict = sorted(kernel_dict.items(), key=lambda x: x[1]['hotspot_num'])

for name, v in sorted_k_dict:
#	print("{}: {}\n\n".format(name, v))
	print("\nHotspot num: {}".format(v['hotspot_num']))
	print("Kernel name: {}".format(k))
	print("Percentage of total exec time: {}".format(v['percent_of_total']))
	print("Total time (ms): {}".format(v['total_time']))
	print("Instances:")
	for instance in v['instances']:
		print("\n- NVTX marker: {}".format(instance.mName))
		kernel_dur_ms = 1.0 * instance.kDuration / 1e6
		print("-- kernel duration: (ms) {}".format(kernel_dur_ms))
		print(instance.mName)
		try:
			nvtx_data = eval(instance.mName)
		except:
			continue
		if nvtx_data['op'] == 'conv2d':
			N, C, H, W = nvtx_data['input_tensor']['shape']
			K, _, R, S = nvtx_data['weight_tensor']['shape']
			PP_H, PP_W = nvtx_data['padding']
			SS_H, SS_W = nvtx_data['stride']
			_, time_s = conv1d_2d_flops(N, C, H, W, K, R, S, PP_H, PP_W, SS_H, SS_W,
						gpu_clock_mhz=1082, num_sms=80,
						tensor_cores_per_sm=8,
						fmas_per_tensor_core=64,
						glob_mem_bandwidth_gb=900)
			time_ms = 1.0 * time_s * 1e3
			print("Kernel dur (ms): {}".format(kernel_dur_ms))
			pct_of_sol = kernel_dur_ms / time_ms
			print("-- Tensor Core SOL time (ms): {}".format(time_ms))
			print("-- Ratio of actual to Tensor Core SOL time: {}".format(pct_of_sol))


# {'op': 'conv2d', 'input_tensor': {'shape': (1, 1, 32, 32), 'type': 'float32'}, 'weight_tensor': {'shape': (6, 1, 5, 5), 'type': 'float32'}, 'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'groups': 1}
		


# N = batch size
# C = number of input channels
# H = height
# W = width
# K = number of output channels
# R = filter height
# S = filter width
# PP_H = padding for height (symmetric padding assumed)
# PP_W = padding for width (symmetric padding assumed)
# SS_H = stride for height (symmetric stride assumed)
# SS_W = stride for width (symmetric stride assumed)
#def conv1d_2d_flops(N, C, H, W, K, R, S, PP_H, PP_W, SS_H, SS_W,
#                    gpu_clock_mhz=1082,
#                    num_sms=80, tensor_cores_per_sm=8, fmas_per_tensor_core=64,
#                    glob_mem_bandwidth_gb=900):

#for hs_name in hs_names:
#	if hs_name in kernel_dict:
#		print(kernel_dict[hs_name]) 

#	if sys.stdout.isatty():
#
#		print(
#
#			colored(kernel.kName, 'green'),
#
#			colored(kernel.mName, 'red')
#
#		)
#
#	else:
#
#		print(kernel.kName, kernel.mName)



#Print info

#for kernel in Kernel.kernels:

#	print(kernel.__dict__)



db.close()
