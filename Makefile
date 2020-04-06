all: collision tvl1


collision:
	cd collision_mask; mkdir build; cd build; cmake ..; make 

tvl1:
	cd tvl1flow; make
