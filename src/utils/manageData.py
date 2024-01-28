import os
import shutil

def deleteAllOcurrancesOfDirectory(directoryPath, target ):
	for root, folders, files in os.walk(directoryPath):
			for folder in folders:
				if(os.path.splitext(folder)[0] == target):
					shutil.rmtree(os.path.join(root, folder))


if __name__ == "__main__":
	deleteAllOcurrancesOfDirectory("data/babyslakh_16k","training_data_drums")