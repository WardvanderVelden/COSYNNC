activationFunction = 'relu';
layerDepth = 3;

stateSpaceEta = [0.1, 0.1];
stateSpaceLowerBound = [-5, -10];
stateSpaceUpperBound = [5, 10];

inputSpaceEta = [1000];
inputSpaceLowerBound = [0];
inputSpaceUpperBound = [5000];

w0 = [
	[-0.0891324058, -0.032202974],
	[-1.6052928, -0.99412334],
	[-4.04514217, -2.0220499],
	[7.58031607, 3.88431883],
	[-0.152679235, 0.00321149593],
	[-0.0579421148, 0.0443104431],
	[1.44097364, 0.726851583],
	[0.0204159785, 0.0147727132],
];
b0 = [-0.0128270164, 0.733252287, 1.74178731, -2.64522314, -0.00324068731, -0.0428531058, -0.49076876, -0.0350604095];
w1 = [
	[-0.0936322138, 0.0567288473, -0.0670611709, 5.2601099e-06, 0.0242956728, -0.0899259895, 0.0154457241, 0.0398196056],
	[-0.0524214357, 0.0822654516, 0.0538833104, -0.0465585925, 0.0227931887, 0.0358181223, 0.00711458456, 0.0728562847],
	[0.0179819912, 0.600321233, 0.599719942, 3.65561128, -0.0980429351, 0.00852091331, 0.671674311, -0.0552833602],
	[-0.0580312498, -0.0555113554, -0.0627613962, -0.0562501252, 0.0888744816, 0.0139147043, 0.0479101613, -0.00957819074],
	[-0.00190823525, 0.184637532, 0.0323903635, 0.692666948, -0.0593123175, -0.082565479, 0.0357904471, -0.0885198712],
	[-0.0131166726, -1.26415956, -3.11822104, 0.86281985, 0.0380496755, 0.0940705016, 0.186584458, 0.0235304981],
	[-0.0640792698, 0.0867027566, -0.0323798768, 1.73636818, -0.11448402, 0.0404228754, 0.383745402, -0.00455963006],
	[-0.0092606321, 0.26436162, 0.229758546, 1.64039075, 0.0479930826, 0.0392930135, 0.412521809, -0.00520251831],
];
b1 = [-0.0799546242, -0.115221985, -2.63740754, -0.0295376629, -0.478349745, 1.25962174, -1.41683757, -1.17997003];
w2 = [
	[-0.0566206053, -0.00440450897, 1.45867336, 0.0812836215, 0.234643057, -1.27707183, 0.662004471, 0.617099702],
	[0.0516757295, 0.07192453, -1.46205568, 0.0808850005, -0.305284619, 1.23366952, -0.740209877, -0.706317663],
];
b2 = [2.47156048, -2.46073771];
