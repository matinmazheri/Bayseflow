simulator.sample(N)

It generates a dict with the following formation

output = {
"param_name_1" : numpy (N,1)
...
"param_name_n" : numpy (N,1)

"output_simulator_1_name" : numpy (N,d_1) d_1 is the shape of the output_1 of simulator 

"output_simulator_1_name" : numpy (N,d_n) d_n is the shape of the output_n of simulator 


"meta_variable_1" : numpy : (d_1)

"meta_variable_n" : numpy : (d_n)
}

This output can be fed into approximiator.sampel(output, NumberOfSamples)

The output of posterior network is 

{

"param_name_1" : numpy (N,NumberOfSamples,1)
...
"param_name_n" : numpy (N,NumberOfSamples,1)

}

The normalizing flow take input N test samples o the shape (d_1,d_2,...,d_n) aftering being processed by adaptor and summary network and return NumberOfSamples samples of parameters. For each parameer we have (N,NumberOfSamples,1) numpy array.