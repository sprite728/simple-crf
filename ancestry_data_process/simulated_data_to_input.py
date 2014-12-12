import numpy

WindowSize=10000

def read_neanderthal_allele(file):
	f=open(file,'r')
	neanderthal_allele=[]
	for line in f:
		neanderthal_allele.append(int(line.strip()))
	return neanderthal_allele

def read_ancestral_pop_allele(file):
	f=open(file,'r')
	ancestral_allele=[]
	for line in f:
		ancestral_allele.append(map(int,list(line.strip())))
	return ancestral_allele

def read_pos(file):
	position=[]
	f=open(file,'r')
	for line in f:
		line =line.rstrip().split('\t')
		position.append(line[3])
	return position

def split_into_winow(position):
	start = 0
	index = 0
	position_index=[]
	window_range = []
	start_id = 0
	end_id = 0
	for i in range(len(position)):
		if position[i]-start<=WindowSize:
			position_index.append(index)
			end_id = i
		else:
			window_range.append((start_id,end_id))
			index +=1
			start=WindowSize*(index)
			start_id=i+1
	if position[i]-start<=WindowSize:
		window_range.append((start_id,end_id))
	print len(window_range),len(position_index),position_index[-1]
	print window_range
	return position_index,window_range

def generate_feature_matrix(neanderthal_allele,ancestral_allele,admixed_allele):
	feature=zeros(len(neanderthal_allele),4*len(admixed_allele[0]))
	for i in range(len(neanderthal_allele)):
		allele_count=[ancestral_allele[i].count(0),ancestral_allele[i].count(1)]
		if allele_count[1-neanderthal_allele[i]]==sum(allele_count):   # Neanderthal allele != African allele
			for j in range(len(admixed_allele[i])):
				index = 4*j
				if admixed_allele[i][j]==neanderthal_allele[i]:
					feature[i][index+0]=1	# means positive
					feature[i][index+1]=2	# means negative
				else:
					feature[i][index+0]=2
					feature[i][index+1]=1
								
if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name",default='out',help="output file prefix")
	neanderthal_allele=read_neanderthal_allele(args.name+'.2.geno') # Neanderthal allele
	ancestral_allele=read_ancestral_pop_allele(args.name+'.1.geno')	# African allele
	admixed_allele=read_ancestral_pop_allele(args.name+'.ADMIXED.geno') # European allele
	position=read_pos(args.name+'.snp')
	position_index,window_range=split_into_winow(position)
	