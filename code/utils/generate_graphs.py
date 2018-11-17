from os import listdir
import sys
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def makeStatsFile(fold):
	

	file = open ('threshold_stats' + str(fold) + '.txt' , 'r')
	#print ("Fold = " + fold)
	if (fold == '1'):
		output_file = open ('stats_file' + '.txt' ,'w')
		output_file.write("Fold,Threshold,Accuracy,Error,cm00,cm01,cm10,cm11\n")
	else:
		output_file = open('stats_file' + '.txt' ,'a')		
	
	for line in file:
		threshold = float(line.split(' ')[0])
		accuracy = float(line.split(' ')[1])
		error = 1.0 - float(line.split(' ')[1])
		cm00 = line.split(' ')[2].split('[')[1].split(',')[0]
		cm01 = line.split(' ')[3].split(',')[0]
		cm11 = line.split(' ')[4].split(',')[0]
		cm10 = line.split(' ')[5].split(']')[0]
		output_file.write(str(fold) + ',' + str(threshold) + ',' + str(accuracy) + ',' + str(error) + ',' + str(cm00) + ',' + str(cm01) + ',' + str(cm10) + ',' + str(cm11) + '\n')
	file.close()
	output_file.close()


def makeStatsFiles(fold):
	file = open ('threshold_stats' + str(fold) + '.txt' , 'r')
	#print ("Fold = " + fold)
	output_file = open('stats_file' + str(fold)+ '.txt' ,'w')		
	output_file.write("Fold,Threshold,Accuracy,Error,cm00,cm01,cm10,cm11\n")
	for line in file:
		threshold = float(line.split(' ')[0])
		accuracy = float(line.split(' ')[1])
		error = 1.0 - float(line.split(' ')[1])
		cm00 = line.split(' ')[2].split('[')[1].split(',')[0]
		cm01 = line.split(' ')[3].split(',')[0]
		cm11 = line.split(' ')[4].split(',')[0]
		cm10 = line.split(' ')[5].split(']')[0]
		output_file.write(str(fold) + ',' + str(threshold) + ',' + str(accuracy) + ',' + str(error) + ',' + str(cm00) + ',' + str(cm01) + ',' + str(cm10) + ',' + str(cm11) + '\n')
	file.close()
	output_file.close()




def main ():

	
	
	
	if (str(sys.argv[1]) == 'boxplot'):
		makeStatsFile("1")
		makeStatsFile("2")
		makeStatsFile("3")
		makeStatsFile("4")
		makeStatsFile("5")
		df = pd.read_csv('stats_file.txt', delimiter = ',')
		sns.boxplot(x="Fold", y="Accuracy", data=df)
		sns.despine(offset=10, trim=True)
		print ("Saving....")
		plt.savefig('boxplot.png')
		print ("Saved....")
	
	if (str(sys.argv[1]) == 'lineplot'):	
		# makeStatsFiles("1")
		# makeStatsFiles("2")
		# makeStatsFiles("3")
		# makeStatsFiles("4")
		# makeStatsFiles("5")

		# df1 = pd.read_csv('stats_file1.txt', delimiter = ',')
		# df2 = pd.read_csv('stats_file2.txt', delimiter = ',')
		# df3 = pd.read_csv('stats_file3.txt', delimiter = ',')
		# df4 = pd.read_csv('stats_file4.txt', delimiter = ',')
		# df5 = pd.read_csv('stats_file5.txt', delimiter = ',')

		makeStatsFile("1")
		makeStatsFile("2")
		makeStatsFile("3")
		makeStatsFile("4")
		makeStatsFile("5")
		df = pd.read_csv('stats_file.txt', delimiter = ',')

		print ("Saving....")
		colors = ["blue", "black", "yellow", "green", "purple"]
		g = sns.lineplot (x = 'Threshold', y = 'Accuracy', hue = 'Fold', palette = colors, data = df, markers = True)
		
		



		plt.savefig('lineplot.png')
		print ("Saved....")

	if (str(sys.argv[1]) == 'lineplots'):
		makeStatsFiles("5")		
		df = pd.read_csv('stats_file5.txt', delimiter = ',')
		g = sns.lineplot (x = 'Threshold', y = 'Accuracy', data = df, markers = True)
		plt.savefig('lineplot5.png')
	"""

	df = pd.read_csv('threshold_stats1.txt', delimiter=' ')
	#print (df)
	#sns.lineplot(x='Threshold', y='Accuracy', data=df)
	
	sns.set(style="darkgrid")
	g = sns.FacetGrid(df, row="Accuracy", col="Threshold", margin_titles=True)
	#bins = np.linspace(0, 60, 13)
	#g.map(plt.hist, "Accuracy", color="steelblue", bins=bins)

	
	"""


if __name__ == "__main__":
    main()