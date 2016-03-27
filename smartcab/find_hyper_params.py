import agent

if __name__ == '__main__':
	# Perform exhaustive search over all paramter combinations
	# Save parameters and scores into csv file for easy sorting and plotting

	# Declare parameter ranges to search over
	sigmoid_offset_range = [2., 4., 6., 8.]
	sigmoid_rate_range   = [10**i for i in range(-4, 0)]
	alpha_decay_range    = [0.1, 0.5, 0.9, 1.3]
	gamma_range          = [0.1, 0.5, 0.9, 1.3]

	csv_string = 'sigmoid_offset,sigmoid_rate,alpha_decay,gamma,score\n'
	score = -1.

	# Sweep over all paramter combinations
	for sigmoid_offset in sigmoid_offset_range:
		for sigmoid_rate in sigmoid_rate_range:
			for alpha_decay in alpha_decay_range:
				for gamma in gamma_range:
					# Run trials with given parameter combination
					score = agent.run(sigmoid_offset=sigmoid_offset, sigmoid_rate=sigmoid_rate, alpha_decay=alpha_decay, gamma=gamma)

					csv_string += '%f,%f,%f,%f,%.4f\n' % (sigmoid_offset,sigmoid_rate,alpha_decay,gamma,score)
					print 'Hyper-parameter search status: %f,%f,%f,%f,%.4f' % (sigmoid_offset,sigmoid_rate,alpha_decay,gamma,score)  # [debug]

	# Write results to hyper_params.csv
	fo = open('hyper_params.csv', 'wb')
	fo.write(csv_string);
	fo.close()
