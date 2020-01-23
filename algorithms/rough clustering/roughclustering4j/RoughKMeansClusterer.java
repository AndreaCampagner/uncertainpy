package roughclustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.OptionalDouble;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implements a Rough KMeans rough clusterer
 * @author Andrea Campagner
 *
 */
public class RoughKMeansClusterer extends RoughClusterer {
	
	private long seed = -1;
	
	/**
	 * Construct a Rough KMeans rough clusterer
	 * @param k, number of clusters
	 * @param iterations, number of iterations
	 * @param threshold, threshold for insertion into clusters
	 * @param wu, weight of the upper region
	 * @param wl, weight of the lower region
	 * @param restarts, number of restarts
	 * @param seed, seed for the initialization
	 */
	public RoughKMeansClusterer(int k, int iterations, double threshold, double wu, double wl, int restarts, long seed) throws Exception {
		super();
		this.k = k;
		this.iterations = iterations;
		this.threshold = threshold;
		this.wu = wu;
		this.wl = wl;
		this.restarts = (restarts > 0)? restarts : 1;
		this.seed = seed;
	}
	
	/**
	 * Construct a Rough KMeans rough clusterer
	 * @param k, number of clusters
	 * @param iterations, number of iterations
	 * @param threshold, threshold for insertion into clusters
	 * @param wu, weight of the upper region
	 * @param wl, weight of the lower region
	 * @param restarts, number of restarts
	 */
	public RoughKMeansClusterer(int k, int iterations, double threshold, double wu, double wl, int restarts) throws Exception {
		this(k, iterations, threshold, wu, wl, restarts, 0);
	}
	
	
	@Override
	public void buildClusterer(Instances data) throws Exception{
		for(int r = 0; r < restarts; r++){
		int card = data.numInstances();
		ArrayList<ArrayList<Integer>> clustering = new ArrayList<ArrayList<Integer>>(
				Collections.nCopies(card, new ArrayList<Integer>()));
		
		//Set the cluster representatives
		weights = new double[data.numAttributes()];
		for(int i = 0; i < weights.length; i++)
			weights[i] = 1.0/(data.numAttributes() - 1);
		Instance[] centroids;
		centroids = setSeed(data,seed);
		if(r == 0)
		this.centroids = Arrays.copyOf(centroids, k);
		
		//Foreach iteration
		for(int i = 0; i < iterations; i++){
			//Compute the rough clustering
			for(int instInd = 0; instInd < data.numInstances(); instInd++){
				Instance inst = new DenseInstance(data.get(instInd));
				double[] dists = new double[k];
				double minDist = Double.MAX_VALUE;
				clustering.set(instInd, new ArrayList<Integer>());
				
				for(int j = 0; j < k; j++){
					dists[j] = computeDistance(data, inst, centroids[j], weights);
					//System.out.println(dists[j]);
					if(dists[j] < minDist)
						minDist = dists[j];
				}
				//System.out.println(minDist);
				for(int j = 0; j < k; j++){
					if(dists[j] == minDist || minDist/dists[j] >= threshold)
						clustering.get(instInd).add(j);
				}
			}
			
			//Build the orthopartition
			Orthopartition pi = new Orthopartition(clustering, data);
			
			//Recompute the representatives using the weighted mean or mode
			for(int j = 0; j < pi.getFamily().size(); j++){
				
				if(pi.getFamily().get(j).isEmpty()){
					continue;
				}
				
				List<Instance> lower = new ArrayList<Instance>();
				List<Instance> upper = new ArrayList<Instance>();
				for(int instInd = 0; instInd < data.numInstances(); instInd++){
					if(clustering.get(instInd).contains(j)){
						if(clustering.get(instInd).size() == 1)
							lower.add(data.get(instInd));
						upper.add(data.get(instInd));
					}		
				}
				if(lower.size() == upper.size()){
					for(int a = 0; a < data.numAttributes(); a++){
							final int fA = a;
							if(data.attribute(a).isNumeric()){//if numeric compute mean
								OptionalDouble avg = upper.stream().mapToDouble(inst -> inst.value(fA)).average();
								centroids[j].setValue(a, avg.getAsDouble());
								}else{//if discrete compute mode
									centroids[j].setValue(a, weightedMode(data, a, pi.getFamily().get(j), 0, 1));
								}
					}
				}else if(lower.size() == 0 && upper.size() != 0){
					for(int a = 0; a < data.numAttributes(); a++){
							final int fA = a;
							if(data.attribute(a).isNumeric()){//if numeric compute mean
							OptionalDouble avg = upper.stream().mapToDouble(inst -> inst.value(fA)).average();
							centroids[j].setValue(a, avg.getAsDouble());
							}else{//if discrete compute mode
								centroids[j].setValue(a, weightedMode(data, a, pi.getFamily().get(j), 1, 0));
							}
					}
				} else if (upper.size() != 0){
					for(int a = 0; a < data.numAttributes(); a++){
							final int fA = a;
							if(data.attribute(a).isNumeric()){//if numeric compute mean
								OptionalDouble avgL = lower.stream().mapToDouble(inst -> inst.value(fA)).average();
								OptionalDouble avgU = upper.stream().mapToDouble(inst -> inst.value(fA)).average();
								centroids[j].setValue(a, wl*avgL.getAsDouble() + wu*avgU.getAsDouble());
							}else{//if discrete compute mode
								centroids[j].setValue(a, weightedMode(data, a, pi.getFamily().get(j), wu, wl));
							}
					}
				}
			}
			//Recomputes the weight
			if(reweight)
				weightAttributes(data, new Orthopartition(clustering, data));
			//If the new clustering is better than the old substitute the representatives
			if(computeDaviesBouldin(data, pi, this.centroids, weights) > computeDaviesBouldin(data, pi, centroids, weights) || o == null){
				this.centroids = Arrays.copyOf(centroids, centroids.length);
				o = pi;
			}
		}
		}
	}
}
