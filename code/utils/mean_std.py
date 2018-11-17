import statistics


def main ():

	coffee_file = open ('coffee.txt', 'r')
	not_coffee_file = open ('not_coffee.txt', 'r')

	threshold_coffee_list = []
	threshold_not_coffee_list = []
	for line in coffee_file:
		threshold_coffee_list.append(float(line))

	for line in not_coffee_file:
		threshold_not_coffee_list.append(float(line))


	#print ("THRESHOLD_COFFEE_LIST: " + str(threshold_coffee_list))
	#print ("THRESHOLD_NOT_COFFEE_LIST: " + str(threshold_not_coffee_list))

	coffee_threshold_mean = reduce(lambda x, y: x + y, threshold_coffee_list) / len(threshold_coffee_list)
	not_coffee_threshold_mean = reduce(lambda x, y: x + y, threshold_not_coffee_list) / len(threshold_not_coffee_list)
	coffee_threshold_std = statistics.stdev(threshold_coffee_list)
	not_coffee_threshold_std = statistics.stdev(threshold_not_coffee_list)

	print ("COFFEE: ")
	print ("MEAN: " + str(coffee_threshold_mean))
	print ("STD: " + str(coffee_threshold_std))

	print ("NOT COFFEE: ")
	print ("MEAN: " + str(not_coffee_threshold_mean))
	print ("STD: " + str(not_coffee_threshold_std))






















if __name__ == "__main__":
    main()