package roughclustering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implements a generic Rough Clusterer abstract interface
 * @author Andrea Campagner
 */
public abstract class RoughClusterer{
	
	
	protected Instance[] centroids;
	protected int k;
	protected int iterations;
	protected double threshold;
	protected double wu;
	protected double wl;
	protected double[] weights;
	protected int restarts;
	protected boolean useHeuristic = true;
	protected boolean reweight = true;
	protected Orthopartition o = null;

	public Orthopartition getClustering(){
		return o;
	}
	
	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public double getWu() {
		return wu;
	}

	public void setWu(double wu) {
		this.wu = wu;
	}

	public double getWl() {
		return wl;
	}

	public void setWl(double wl) {
		this.wl = wl;
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}

	public int getRestarts() {
		return restarts;
	}

	public void setRestarts(int restarts) {
		this.restarts = restarts;
	}

	public boolean isUseHeuristic() {
		return useHeuristic;
	}

	public void setUseHeuristic(boolean useHeuristic) {
		this.useHeuristic = useHeuristic;
	}

	public boolean isReweight() {
		return reweight;
	}

	public void setReweight(boolean reweight) {
		this.reweight = reweight;
	}

	/**
	 * Compute the rough clustering
	 * @param data, dataset
	 * @throws Exception
	 */
	public abstract void buildClusterer(Instances data) throws Exception;
	
	/**
	 * Compute the best assignment of the given instance
	 * @param inst, instance
	 * @param data, dataset
	 * @return the index of the best assignment
	 * @throws Exception
	 */
	public int clusterInstance(Instance inst, Instances data) throws Exception{
		double[] dists = new double[k];
		double minDist = Double.MAX_VALUE;
		int ind = -1;
		for(int j = 0; j < k; j++){
			dists[j] = computeDistance(data, inst, centroids[j], weights);
			if(dists[j] < minDist){
				minDist = dists[j];
				ind = j;
			}
		}
		return ind;
	}
	
	/**
	 * Compute the assignment of the given instances
	 * @param data, dataset
	 * @return the resulting orthopartition
	 * @throws Exception
	 */
	public Orthopartition clusterInstances(Instances data) throws Exception{
		ArrayList<ArrayList<Integer>> clustering = new ArrayList<ArrayList<Integer>>(
				Collections.nCopies(data.numInstances(), new ArrayList<Integer>()));
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
		return new Orthopartition(clustering, data);
	}
	
	/**
	 * Recomputes the weights of the attributes
	 * @param data, dataset
	 * @param o, an orthopartition
	 * @throws Exception
	 */
	protected void weightAttributes(Instances data, Orthopartition o) throws Exception{
		for(int i = 0; i < weights.length; i++){
				ArrayList<ArrayList<Integer>> clustering = new ArrayList<ArrayList<Integer>>(
						Collections.nCopies(data.numInstances(), new ArrayList<Integer>()));
				Orthopartition p;
				ArrayList<Orthopair> family = new ArrayList<Orthopair>();
				//Computes orthocovering determined by the current (numeric) attribute
				if(data.attribute(i).isNumeric()){
					for(int j = 0; j < data.numInstances(); j++){
						double max = data.attributeStats(i).numericStats.max;
						double min = data.attributeStats(i).numericStats.min;
						HashSet<Instance> pos = new HashSet<Instance>();
						HashSet<Instance> neg = new HashSet<Instance>();
						for(int k = 0; k < data.numInstances(); k++)
							if(j==k || 1 - (Math.abs(data.get(j).value(i) - data.get(k).value(i))/(max - min)) >= threshold)
								pos.add(data.get(k));
							else
								neg.add(data.get(k));
						Orthopair op = new Orthopair(neg, pos, new HashSet<Instance>());
						family.add(op);
					}
					//If useHeuristic then compacts the orthocovering
					if(useHeuristic){
						ArrayList<Orthopair> tf = new ArrayList<Orthopair>();
						HashSet<Instance> s = new HashSet<Instance>();
						while(!s.equals(family.get(0).getUniverse())){
							int max = 0;
							int im = -1;
							for(int io = 0; io < family.size(); io++){
								Orthopair to = new Orthopair(family.get(io));
								to.getP().addAll(s);
								if(to.getP().size() > max){
									max = to.getP().size();
									im = io;
								}
							}
							s.addAll(family.get(im).getP());
							tf.add(family.get(im));
						}
						family = tf;
					}
					boolean overlap = false;
					for(Orthopair o1 : family){
						for(Orthopair o2: family){
							Orthopair tmp = new Orthopair(o1);
							if(o1 != o2 && !tmp.intersect(o2).isEmpty()){
								overlap = true;
								break;
							}
						}
						if(overlap)
							break;
					}
					p = new Orthopartition(family, overlap);
				}else{ //Compute the orthopartition determined by the current (discrete) attribute
					for(int j = 0; j < data.numInstances(); j++){
						clustering.get(j).add((int) data.get(j).value(i));
					}
					p = new Orthopartition(clustering, data);
				}
				weights[i] = o.mutualInformation(p);
		}
		double sum = 0;
		for(int i = 0; i < weights.length; i++)
			sum += weights[i];
		for(int i = 0; i < weights.length; i++)
			weights[i] /= sum;
		}
	
	
	/**
	 * Set the initial seed centroids/cluster representatives
	 * @param data, dataset
	 * @return
	 */
	protected Instance[] setSeed(Instances data, long seed){
		Instance[] centroids = new Instance[k];
		Random r = new Random(seed);
		int randomNum = r.nextInt(data.numInstances());
		centroids[0] = data.get(randomNum);
		ArrayList<Integer> off = new ArrayList<Integer>();
		off.add(randomNum);
		
		//At each iteration selects as new representative the instance with the maximum distance
		//w.r.t. the already selected representatives
		for(int i = 1; i < k; i++){
			double maxDist = 0;
			int best = -1;
			for(int inst = 0; inst < data.numInstances(); inst++){
				if(off.contains(inst))
					continue;
				double dist = 0;
				for(int j = 0; j < i; j++){
				  dist += computeDistance(data, centroids[j], data.get(inst), weights);
				}
				dist /= i;
				if(dist > maxDist){
					maxDist = dist;
					best = inst;
				}
			}
			centroids[i] = data.get(best);
			off.add(best);
		}
		return centroids;
	}
	
	/**
	 * Compute the distance between two instances
	 * @param data, dataset
	 * @param x, instance
	 * @param y, instance
	 * @param weights
	 * @return the distance d(x,y)
	 */
	protected double computeDistance(Instances data, Instance x, Instance y, double[] weights){
		double dist =  0 ;
		for(int a = 0; a < data.numAttributes(); a++){
				double value = 0;
				if(data.attribute(a).isNumeric()){
					double max = data.attributeStats(a).numericStats.max;
					double min = data.attributeStats(a).numericStats.min;
					value = weights[a]*
							(Math.abs(x.value(a) - y.value(a))/(max - min));
				}else{//attribute is discrete
					value = weights[a]*((x.value(a) == y.value(a))? 0 : 1);
				}
				dist += value;
		}
		return dist;
	}
	
	/**
	 * Compute Davis Bouldin index (DB-index)
	 * @param data, dataset
	 * @param o, orthopartition
	 * @param centroids
	 * @param weights, 
	 * @return the value of the DB-index
	 */
	protected double computeDaviesBouldin(Instances data, Orthopartition o, Instance[] centroids, double[] weights){
		double db = 0;
		double[] S = new double[centroids.length];
		double[][] D = new double[centroids.length][centroids.length];
		
		for(int j = 0; j < o.getFamily().size(); j++){
			//Foreach orthopair in the collection compute its compactness
			D[j][j] = 0;
			S[j] = 0;
			HashSet<Instance> P = o.getFamily().get(j).getP();
			HashSet<Instance> Bnd = o.getFamily().get(j).getBnd();
			if(Bnd.isEmpty() || P.isEmpty()){
				HashSet<Instance> tmp = new HashSet<Instance>(P);
				tmp.addAll(Bnd);
				for(Instance i : tmp)
					S[j] += computeDistance(data, i, centroids[j], weights);
				S[j] /= tmp.size();
			}else{
				double SP = 0, SB = 0;
				for(Instance i : P)
					SP += wl*computeDistance(data, i, centroids[j], weights);
				for(Instance i : Bnd)
					SB += wu*computeDistance(data, i, centroids[j], weights);
				S[j] = SP/P.size() + SB/Bnd.size();
			}
			
			//Foreach other orthopair in the collection compute the distance between the representatives
			for(int k = j + 1; k < o.getFamily().size(); k++){
				D[j][k] = computeDistance(data, centroids[j], centroids[k], weights);
				D[k][j] = D[j][k];
			}
		}
		
		//Compute the coefficient
		for(int j = 0; j < o.getFamily().size(); j++){
			double maxCoeff = 0;
			for(int k = 0; k < o.getFamily().size(); k++)
				if(k != j){
					double tmp = (S[j] + S[k])/D[j][k];
					if(tmp > maxCoeff)
						maxCoeff = tmp;
				}
			db += maxCoeff;
		}
		return db/centroids.length;
		
	}
	
	/**
	 * Compute the weighted mode
	 * @param data, dataset
	 * @param numAttr, index of the attribute to be considered
	 * @param o, orthopartition
	 * @param wu, upper region weight
	 * @param wl, lower region weight
	 * @return the index of the weighted mode
	 */
	protected static double weightedMode(Instances data, int numAttr, Orthopair o, double wu, double wl) {
		
		Enumeration vals = data.attribute(numAttr).enumerateValues();
		HashMap<String, Double> map = new HashMap<String, Double>();
		
		while(vals.hasMoreElements()) {
			map.put(vals.nextElement().toString(), 0.0);
		}
		
		for(Instance i : o.getP()){
			String v = String.valueOf(i.stringValue(numAttr));
			double d = map.get(v);
			d += wl + d;
			map.replace(v, d);
		}
		
		for(Instance i : o.getBnd()){
			String v = String.valueOf(i.stringValue(numAttr));
			double d = map.get(v);
			d += wu + d;
			map.replace(v, d);
		}
		
		String max = (String) map.keySet().toArray()[0];
		double count = 0;
		for(String s : map.keySet()){
			if(map.get(s) > count){
				count = map.get(s);
				max = s;
			}
		}
		return (double) data.attribute(numAttr).indexOfValue(max);
	}

}
