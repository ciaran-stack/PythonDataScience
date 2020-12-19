
# open connection to the file company.txt from edgar index
sec_file = open(r'company.txt')

# create new file and write each line to file
cik_file = open("cik.txt", "w+")


# for loop to process each line in file
for line in sec_file:

    # parse the string of of CIK numbers line by line to new cik.txt file
    cik_file.write(line[74:84])

# close file connection
cik_file.close()

