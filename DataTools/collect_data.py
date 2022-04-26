import sys
sys.path.append('MacAPI')
import os
import numpy as np
import matplotlib.pyplot as pt
import time
import h5py
import sim
import math
import argparse as ap
import cv2


def setNumberOfBlocks(clientID,blocks,typeOf,mass,blockLength,frictionCube,frictionCup):
        '''
        Function to set the number of blocks in the simulation
        '''
        emptyBuff = bytearray()
        res,retInts,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(clientID,'Table',sim.sim_scripttype_childscript,'setNumberOfBlocks',[blocks],[mass,blockLength,frictionCube,frictionCup],[typeOf],emptyBuff,sim.simx_opmode_blocking)
        if res==sim.simx_return_ok:
            print ('Results: ',retStrings) # display the reply from CoppeliaSim (in this case, the handle of the created dummy)
        else:
            print ('Remote function call failed')

def triggerSim(clientID):
    e = sim.simxSynchronousTrigger(clientID)
    print('synct',e)

if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument("shape_type", help="type of shape",choices={'cube','ball','cylinder'})
    parser.add_argument("input_path", help="input path to save trials")
    parser.add_argument("--sim_port",help="simulation port", choices={19990,19991,19992,19993},type=int)
    parser.add_argument("--friction_cube",help="coefficient of friction for cubes",type=float)
    args = parser.parse_args()

    wd = os.getcwd()

#############################################
############coppelia part###################
############################################

    startTimeGeneral = time.perf_counter()

    initialCount = np.array([25,30,35,40,45,50])
    targetCount = np.array([2,3,4,5,6,7,8,9,10])
    forwardVel = np.array([5,10,15,20,25,30]) * np.pi/180 * -1
    backwardVel = np.array([30,40,50,60]) * np.pi/180
    pctgForce = np.array([0.3,0.4,0.5,0.6,0.7])
    rangeAngle = np.array([75,77.5,80,82.5,85,87.5,90]) * -1

    for trial in range(1):

        sim.simxFinish(-1)      # just in case, close all opened connections
        clientID = sim.simxStart('127.0.0.1', args.sim_port if args.sim_port else 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
        if clientID != -1:
            print('Connected to remote API server')
        else:
            print("fail")
            sys.exit()
        
        returnCode = sim.simxSynchronous(clientID,True)
        returnCode = sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)
        retrunCode = sim.simxSetBoolParam(clientID,sim.sim_boolparam_display_enabled,False,sim.simx_opmode_oneshot)
        if returnCode!=0 and returnCode!=1:
            print("something is wrong")
            print(returnCode)
            exit(0)
        
        triggerSim(clientID)
        
        # init the finishing signal
        returnCode, signalValue=sim.simxGetFloatSignal(clientID,'toPython',sim.simx_opmode_streaming)
        
        # init the force sensor reading
        res, f = sim.simxGetObjectHandle(clientID, 'Force_sensor', sim.simx_opmode_blocking)   # sensor under the cup
        returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_streaming)

        res, box = sim.simxGetObjectHandle(clientID, 'box', sim.simx_opmode_blocking)    # senor under the box
        returnCode1, state1, forceVector1, torqueVector1 = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_streaming)

        # get the handle for the source container
        res, pour = sim.simxGetObjectHandle(clientID, 'joint', sim.simx_opmode_blocking)

        # get the handle for the vision sensors
        res, camRGB = sim.simxGetObjectHandle(clientID, 'rgb', sim.simx_opmode_blocking)
        res, camDepth = sim.simxGetObjectHandle(clientID, 'depth', sim.simx_opmode_blocking)
        res, camFixed = sim.simxGetObjectHandle(clientID, 'fixed', sim.simx_opmode_blocking)
        res, camSide = sim.simxGetObjectHandle(clientID, 'sideCam', sim.simx_opmode_blocking)
        
        # height and diameter are in mm
        H = 186.45
        D = 133.33
        frictionCube=args.friction_cube if args.friction_cube else 0.06
        frictionCup=0.8
        #Ice Cube 1 inch
        length=0.025
        massOfBlock = 14.375e-03
        single_block_weight = massOfBlock * 9.8 / 4.448

        triggerSim(clientID)
        number_of_blocks = int(np.random.choice(initialCount))
        print('Initial number of blocks=',number_of_blocks)
        setNumberOfBlocks(clientID,number_of_blocks,args.shape_type,massOfBlock,length,frictionCube,frictionCup)

        #Wait until blocks finish dropping
        while True:
            triggerSim(clientID)
            returnCode, signalValue=sim.simxGetFloatSignal(clientID,'toPython',sim.simx_opmode_blocking)
            if signalValue == 99:
                loop = 40
                while loop > 0:
                    triggerSim(clientID)
                    loop -= 1
                break

        print('Total number of blocks in container: ',number_of_blocks)
        target = int(np.random.choice(targetCount))
        print('Target number of blocks=',target)
        total_weight = single_block_weight * number_of_blocks
        target_weight = single_block_weight * target
        
        #First read of vision sensors
        returnCode, resolution, depthImage = sim.simxGetVisionSensorDepthBuffer(clientID,camDepth,sim.simx_opmode_streaming)
        returnCode, resolution, rgbImage = sim.simxGetVisionSensorImage(clientID,camRGB,0,sim.simx_opmode_streaming)
        returnCode, resolution, rgbImageFixed = sim.simxGetVisionSensorImage(clientID,camFixed,0,sim.simx_opmode_streaming)
        returnCode, resolution, rgbImageSide = sim.simxGetVisionSensorImage(clientID,camSide,0,sim.simx_opmode_streaming)
        
        # record the weight of the receiving cup, positive number makes more sense
        triggerSim(clientID)
        returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_buffer)
        receiver_self_weight = -1 * forceVector[2] / 4.448

        #  use the box reading
        # this box_self_weight includes the receiver cup's weight
        returnCode2, state2, forceVector2, torqueVector2 = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
        # 31.8198 is the weight for the box and empty receiving cup
        box_self_weight = -1 * forceVector2[2] / 4.448

        # get the starting position of source
        returnCode, original_position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_streaming)
        returnCode, original_position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
        returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
        
        # construct the 6 parameters array, in the verse order of H, D, f_total, f_2_pour, force reading, angle displacement
        reading = np.empty([1, 1, 6], dtype=float)

        # convert radians to degree N-m to lbf
        reading[:, :, 0] = original_position * 180 / math.pi
        reading[:, :, 1] = 0
        reading[:, :, 2] = total_weight
        reading[:, :, 3] = target_weight
        reading[:, :, 4] = H
        reading[:, :, 5] = D

        #force direction 
        #f_to_pour and f_total are positive. Force sensor reading is negative. 
        #Take joint to -5 degrees
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, -0.05, sim.simx_opmode_oneshot)
        while True:     
            triggerSim(clientID)
            returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
            print('Angle=',position*180/np.pi)      
            if position <= -5*np.pi/180:
                break
        
        i = 0
        time_step = []
        time_step_sim = []
        data=[]
        dataRGBD = []
        dataVideo = []
        dataVideoFixed = []
        dataVideoRGB = []
        forces = []
        final_cubes=0
        longTrial = False
        saveVideo = True
        
        forSpeed = np.random.choice(forwardVel)
        backSpeed = np.random.choice(backwardVel)
        pctg = np.random.choice(pctgForce)
        backAngle = np.random.choice(rangeAngle)
        forwardStep = False
        constantStep = False

        #Dummy command
        sim.simxGetIntegerParameter(clientID, sim.sim_intparam_program_version, sim.simx_opmode_streaming)
        start_time_sim=sim.simxGetLastCmdTime(clientID)

        # loop until reach the desire target value
        while True:
            triggerSim(clientID)
            
            # Make sure simulation step finishes
            returnCode,pingTime=sim.simxGetPingTime(clientID)
            print('Ping time=',pingTime)

            start_time_host=time.perf_counter()

            #Read Vision Sensors
            #Depth
            returnCode, resolution, depthImage = sim.simxGetVisionSensorDepthBuffer(clientID,camDepth,sim.simx_opmode_buffer)
            depthImage = np.array(depthImage).reshape((resolution[0],resolution[1]))
            #RGB
            returnCode, resolution, rgbImage = sim.simxGetVisionSensorImage(clientID,camRGB,0,sim.simx_opmode_buffer)
            rgbImage = np.array(rgbImage,dtype = np.uint8).reshape(resolution[0],resolution[1],3)

            #Get Cube info
            res, cuboid0Handle=sim.simxGetObjectHandle(clientID,"Cuboid0",sim.simx_opmode_blocking)
            returnCode,cuboid_position=sim.simxGetObjectPosition(clientID,cuboid0Handle,-1,sim.simx_opmode_streaming)
            print(cuboid_position)

            #Array to save images
            rgbdImage = np.concatenate((rgbImage,np.expand_dims(depthImage,axis=2)),axis=2)
            dataRGBD.append(rgbdImage)
            
            returnCode, resolution, videoImage = sim.simxGetVisionSensorImage(clientID,camSide,0,sim.simx_opmode_buffer)
            videoImage = np.array(videoImage,dtype = np.uint8).reshape(resolution[0],resolution[1],3)
            dataVideo.append(videoImage)

            returnCode, resolution, videoImageFixed = sim.simxGetVisionSensorImage(clientID,camFixed,0,sim.simx_opmode_buffer)
            videoImageFixed = np.array(videoImageFixed,dtype = np.uint8).reshape(resolution[0],resolution[1],3)
            dataVideoFixed.append(videoImageFixed)
            
            returnCode, resolution, videoRGB = sim.simxGetVisionSensorImage(clientID,camRGB,0,sim.simx_opmode_buffer)
            videoRGB = np.array(videoRGB,dtype = np.uint8).reshape(resolution[0],resolution[1],3)
            dataVideoRGB.append(videoRGB)
            
                
            print('Force sensor reading=',reading[0,0,1])
            if -1 * reading[0,0,1] < pctg*target_weight and position*180/np.pi > backAngle and not forwardStep:
                    speed = forSpeed
            elif -1 * reading[0,0,1] < pctg*target_weight and position*180/np.pi <= backAngle and not constantStep:
                    speed = 0
                    forwardStep = True
            else:
                    speed = backSpeed
                    constantStep = True
            
            #  use the box reading
            returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)

            # returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_buffer)
            # update the reading, only need to change force reading and angle displacement
            # we want to keep the force reading negative, so + the cup weight
            # weight_in_receiver = ((forceVector[2] / 4.448) + receiver_self_weight)

            weight_in_receiver = ((forceVector[2] / 4.448) + box_self_weight)
            #Log forces to filter the signal
            forces.append(weight_in_receiver)

            # get force senor reading and joint position
            returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
            

            print('Angle=',position*180/np.pi)
            print('Velocity=',speed)
            errorCode = sim.simxSetJointTargetVelocity(clientID, pour, speed, sim.simx_opmode_oneshot)
            i += 1

            if i > 1300:
                longTrial = True
                break

            # prevent overshot when rotate back
            if -1*weight_in_receiver > single_block_weight and position > -5 * math.pi / 180:
                    print("no overshot")
                    errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot)
                    final_cubes = round(-1 * weight_in_receiver / single_block_weight)
                    print(final_cubes, "Number of Cubes in receiver:box")
                    break

            # prepare new data for the next iteration
            reading[0, 0, 1] = weight_in_receiver
            reading[0, 0, 0] = position * 180 / math.pi

            #Dummy for getting the simulation time
            sim.simxGetIntegerParameter(clientID, sim.sim_intparam_program_version, sim.simx_opmode_buffer) # Needed, in order to update the last cmd time
            
            # frequency adjustor 
            duration=time.perf_counter()-start_time_host
            currentTimeSim = sim.simxGetLastCmdTime(clientID)
            duration_sim = (currentTimeSim - start_time_sim)/1000
            start_time_sim = currentTimeSim
            print('Time step host=',duration)
            print('Time step simulation=',duration_sim)
            print('Elapsed time=',time.perf_counter() - startTimeGeneral)

            # collect data
            temp=np.squeeze(reading)
            temp=np.append(temp,speed)
            data.append(temp)
            #time_step.append(time.perf_counter()-start_time_host)
            time_step.append(duration)
            time_step_sim.append(duration_sim)

        # if the model doesn't put the cup back to original position after exiting the loop
        triggerSim(clientID)
        returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)
        if position <0:
                returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
                weight_in_receiver = ((forceVector[2] / 4.448) + box_self_weight)
                final_cubes = round(-1 * weight_in_receiver / single_block_weight)
                print("manually rotate back")
                print(final_cubes, "Number of Cubes in receiver:box")
                while position<0:
                        triggerSim(clientID)
                        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0.2, sim.simx_opmode_oneshot_wait)
                        returnCode, position = sim.simxGetJointPosition(clientID, pour, sim.simx_opmode_buffer)

        triggerSim(clientID)
        sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot_wait)

        # calculate how many cubes spilled out
        returnCode1, state1, forceVector1, torqueVector1 = sim.simxReadForceSensor(clientID, box, sim.simx_opmode_buffer)
        returnCode, state, forceVector, torqueVector = sim.simxReadForceSensor(clientID, f, sim.simx_opmode_buffer)

        # cubes in the cup
        weight_in_receiver = ((forceVector[2] / 4.448) + receiver_self_weight)
        cubes_in_cup=round(-1 * weight_in_receiver / single_block_weight)

        # final_cubes is the count of cubes in the box
        print('Final cubes=',final_cubes)
        print('Cubes in cup=',cubes_in_cup)
        include_spill = final_cubes-cubes_in_cup
        
        path = args.input_path 

        file_name='number_of_blocks{}_target{}_result{}_frictionCube{}_frictionCup{}_length{}_spill{}_trial{}'.format(number_of_blocks,target,final_cubes,frictionCube,frictionCup,length,include_spill,str(time.time())[-3:])
        
        testing=os.path.join(path,file_name+".npy")
        rgbdFile=os.path.join(path,file_name+"_rgbd.h5")
        videoFile=os.path.join(path,file_name+"_video.h5")
        videoFileFixed=os.path.join(path,file_name+"_video_fixed.h5")
        videoFileRGB=os.path.join(path,file_name+"_video_rgb.h5")
        
        #Save time series and video data
        with open(testing,'wb') as f:
            np.save(f,np.array(data))
       
        # with h5py.File(rgbdFile, 'w') as hf:
        #     hf.create_dataset("rgbd",  data=np.flip(np.array(dataRGBD)[:,:186,70:186,:],1), compression="gzip")
        
        with h5py.File(videoFile, 'w') as hf:
            hf.create_dataset("video",  data=np.flip(np.array(dataVideo),1), compression="gzip")
        
        # with h5py.File(videoFileFixed, 'w') as hf:
        #     hf.create_dataset("video",  data=np.flip(np.array(dataVideoFixed),1), compression="gzip")

        # with h5py.File(videoFileRGB, 'w') as hf:
        #     hf.create_dataset("video",  data=np.flip(np.array(dataVideoRGB),1), compression="gzip")
        
        triggerSim(clientID)
        errorCode = sim.simxSetJointTargetVelocity(clientID, pour, 0, sim.simx_opmode_oneshot_wait)

        # close the connection to CoppeliaSim
        x = sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
        sim.simxFinish(clientID)
        print('all done, stop sim. {} cubes spilled, {} time'.format(include_spill,len(time_step)))
        #'''



