filename = "logs/output_111.txt"

causal_errors = {'IRM': [], 'ICP': [], 'ERM': [], 'SEM': []}
non_causal_errors = {'IRM': [], 'ICP': [], 'ERM': [], 'SEM': []}

with open(filename, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'chain_hidden' in line:
            splitted_line = line.strip().split(" ")

            if 'ERM' in line:
                key = 'ERM'
            elif 'ICP' in line:
                key = 'ICP'
            elif 'IRM' in line:
                key = 'IRM'
            else:
                key = 'SEM'

            causal_errors[key].append(float(splitted_line[-2]))
            non_causal_errors[key].append(float(splitted_line[-1]))


for key in causal_errors:
    print("\n\nMean Causal Errors for {}: {}".format(
        key, sum(causal_errors[key])/min(len(causal_errors[key]), 1)))
    print("Mean Non Causal Errors for {}: {}".format(
        key, sum(non_causal_errors[key])/min(len(non_causal_errors[key]), 1)))
