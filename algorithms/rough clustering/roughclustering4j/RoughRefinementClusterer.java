package roughclustering;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.stream.Collectors;

import weka.core.Instance;
import weka.core.Instances;

/**
 * Implements a Rough Refinement rough clusterer
 * @author Andrea Campagner
 *
 */
public class RoughRefinementClusterer extends RoughClusterer{

	@Override
	public void buildClusterer(Instances data) throws Exception {
		weights = new double[data.numAttributes()];
		for(int i = 0; i < weights.length; i++)
			weights[i] = 1.0/(data.numAttributes()-1);
		Orthopartition p = null;
		for(int k = 0; k < iterations; k++){
			//Build the orthocovering defined by the instances
			ArrayList<Orthopair> family = new ArrayList<Orthopair>();
			for(int i1 = 0; i1 < data.numInstances(); i1++){
				HashSet<Instance> pos = new HashSet<Instance>();
				HashSet<Instance> neg = new HashSet<Instance>();
				for(int i2 = 0; i2 < data.numInstances(); i2++){
					double value = computeDistance(data, data.get(i1), data.get(i2), weights);
					if(i1==i2 || value <= 1 - threshold)
						pos.add(data.get(i2));
					else
						neg.add(data.get(i2));
				}
				Orthopair op = new Orthopair(neg, pos, new HashSet<Instance>());
				family.add(op);
			}
			//if useHeuristic compact the orthocovering
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
			p = new Orthopartition(family);
			boolean modified = true;
			
			//Merge the clusters according to their overlap
			while(modified){
				modified = false;
				ArrayList<Orthopair> erased = new ArrayList<Orthopair>();
				ArrayList<Orthopair> added = new ArrayList<Orthopair>();
				for(Orthopair oi: p.getFamily()){
					if(!erased.contains(oi)){
						for(Orthopair oj : p.getFamily()){
							if(oi != oj && !erased.contains(oi) && !erased.contains(oj)){
								Orthopair tmp = new Orthopair(oi);
								tmp.getP().addAll(oj.getP());
								tmp.getN().removeAll(oj.getP());
								int Nmeet = oi.intersect(oj).getLowerSize();
								double D = ((double) Nmeet)/(tmp.getLowerSize() - Nmeet);
								if((oi.getP().containsAll(oj.getP()) || D >= threshold)){
									erased.add(oi);
									erased.add(oj);
									added.add(tmp);
									modified = true;
								}
							}
						}
					}
				}
				p.getFamily().removeAll(erased);
				p.getFamily().addAll(added);
			}
			for(int i = 0; i < data.numInstances(); i++){
				final int fI = i;
				ArrayList<Orthopair> tmp = (ArrayList<Orthopair>) family.stream().filter(o -> o.getP().contains(data.get(fI)))
						.collect(Collectors.toList());
				if(tmp.size() != 1)
					for(Orthopair o : family)
						if(tmp.contains(o)){
							o.getP().remove(data.get(i));
							o.getBnd().add(data.get(i));
						}
			}
			
			//Check if there is overlap among the orthopairs
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
			weightAttributes(data, p);
		}
		p.setFamily((ArrayList<Orthopair>) p.getFamily().stream().filter((Orthopair o) -> !o.isEmpty())
				.collect(Collectors.toList()));
		o = p;
		
	}

	/**
	 * Construct a Rough Refinement rough clusterer
	 * @param iterations, number of iterations
	 * @param threshold, threshold for insertion into clusters
	 */
	public RoughRefinementClusterer(int iterations, double threshold) {
		super();
		this.iterations = iterations;
		this.threshold = threshold;
	}

	@Override
	/**
	 * This method is not defined
	 */
	public int clusterInstance(Instance inst, Instances data) throws Exception {
		return -1;
	}

}
