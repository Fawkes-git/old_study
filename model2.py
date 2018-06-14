import serial
import os.path
import time
import csv
import sys

def serial_read(start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    elapsed_time = round(elapsed_time,1)
    if elapsed_time > 10:
      sys.exit()

    bytes_data = ser.readline()
    str_data = bytes_data.decode('utf-8')
    str_data = str_data.rstrip('\r\n')
    str_data = str(elapsed_time) + '\t' + str_data 
    print(str_data)
    write_csv(str_data)


def write_csv(str_data):
    path = os.path.join(os.getcwd(),'train.tsv')
    file = open(path, 'a')
    writer = csv.writer(file, lineterminator='\n')
    csvlist = []
    csvlist.append(str_data)
    writer.writerow(csvlist)
    file.close()

if __name__ == '__main__':
   ser = serial.Serial('/dev/cu.usbmodem1451',9600) #Connection to Arduino
   start_time = time.time()

   path = os.path.join(os.getcwd(),'train.tsv')
   if not(os.path.exists(path)):
      file = open(path, 'w')
      file.write('Time'+'\t'+'Sensor1'+'\t'+'Sensor2'+'\t'+'Sensor3'+'\t'+'Sensor4'+'\t'+'Sensor5'+'\t'+'Sensor6'+'\n')
      file.close()

   while True:
      serial_read(start_time)




