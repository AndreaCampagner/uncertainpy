package roughclustering;

import java.util.ArrayList;
import java.util.HashSet;

import weka.core.Instance;
import weka.core.Instances;


/**
 * Implements an orthopartition
 * Note that all constructors may fail, and throw an exception, if the orthopartition
 * is set to have no overlaps or if the orthopairs are not defined on the same universe
 * @author Andrea Campagner
 *
 */
public class Orthopartition {
	private ArrayList<Orthopair> family;
	boolean overlap;
	
	/**
	 * Construct an orthopartition given a universe a list of lists representation of the orthopartition
	 * @param list, list of lists of instances
	 * @param data, dataset
	 * @throws Exception
	 */
	public Orthopartition(ArrayList<ArrayList<Integer>> list, Instances data) throws Exception{
		overlap = false;
		int numOrthopairs = -1;
		ArrayList<Orthopair> tmp = new ArrayList<Orthopair>();
		for(ArrayList<Integer> l : list)
			for(Integer i : l)
				if(i > numOrthopairs)
					numOrthopairs = i;
		//Assigns the elements in the overlaps to the boundaries
		for(int i = 0; i <= numOrthopairs; i++){
			final int fi = i;
			HashSet<Instance> p = new HashSet<Instance>();
			HashSet<Instance> bnd = new HashSet<Instance>();
			HashSet<Instance> n = new HashSet<Instance>();
			for(int j = 0; j < list.size(); j++){
				if(list.get(j).contains(fi))
					if(list.get(j).size() == 1)
						p.add(data.get(j));
					else
						bnd.add(data.get(j));
				else
					n.add(data.get(j));	
			}
			tmp.add(new Orthopair(n, p, bnd));
		}
		setFamily(tmp);
	}
	
	/**
	 * Support method to convert a list of ints in a list of lists of integers
	 * @param list, a list of integers
	 * @return the corresponding list of lists of integers
	 */
	public static ArrayList<ArrayList<Integer>> convert(int[] list){
		ArrayList<ArrayList<Integer>> tmp = new ArrayList<ArrayList<Integer>>();
		for(int i : list){
			tmp.add(new ArrayList<Integer>());
			tmp.get(tmp.size() - 1).add(i);
		}
		return tmp;
	}
	
	/**
	 * Construct an orthopartition from a list and a universe
	 * @param list, list of cluster assignments for the instances
	 * @param data, dataset
	 * @throws Exception
	 */
	public Orthopartition(int[] list, Instances data) throws Exception{
		this(convert(list), data);
	}

	/**
	 * Construct an orthopartition from a collection of orthopairs and whether the orthopartition should admit overlaps
	 * @param family, a collection of orthopairs
	 * @param overlap, whether the orthopartition admits overlaps (i.e. is an orthocovering)
	 * @throws Exception - the orthopairs are defined on different universes OR orthopairs overlap (and overlap not admitted)
	 */
	public Orthopartition(ArrayList<Orthopair> family, boolean overlap) throws Exception{
		super();
		this.overlap = overlap;
		HashSet<Instance> universe = family.get(0).getUniverse();
		for(Orthopair o : family)
			if(!o.getUniverse().equals(universe))
				throw new Exception ("Not all orthopairs are defined on the same universe");
		if(!overlap){
			//Check if there is an overlap
			for(Orthopair o : family){
				for(Orthopair p: family){
					if(o != p){
						Orthopair tmp1 = new Orthopair(o);
						tmp1.getP().addAll(tmp1.getBnd());
						tmp1.getP().retainAll(p.getP()) ;
						Orthopair tmp2 = new Orthopair(o);
						tmp2.getP().retainAll(p.getBnd());
						if(!tmp1.getP().isEmpty() || !tmp2.getP().isEmpty())
							throw new Exception("Orthopairs overlap");
					}
				}
			}
		}
		this.setFamily(family);
	}
	
	/**
	 * Construct an orthopartition from a collection of orthopairs, automatically compute whether the orthopartition
	 * should admit overlaps.
	 * @param family, a collection of orthopairs
	 * @throws Exception
	 */
	public Orthopartition(ArrayList<Orthopair> family) throws Exception{
		this.setFamily(family);
		overlap = false;
		for(Orthopair o: family){
			for(Orthopair p : family)
				if(o != p){
					Orthopair tmp1 = new Orthopair(o);
					tmp1.getP().addAll(tmp1.getBnd());
					tmp1.getP().retainAll(p.getP()) ;
					Orthopair tmp2 = new Orthopair(o);
					tmp2.getP().retainAll(p.getBnd());
					if(!tmp1.getP().isEmpty() || !tmp2.getP().isEmpty()){
						overlap = true;
						break;
					}
				}
			if(overlap)
				break;
		}
	}
	
	/**
	 * Add an orthopair to the orthopartition.
	 * It may fail if orthopartition is set to have no overlap
	 * @param o, new orthopair
	 * @return whether insertion was successful
	 */
	public boolean addOrthopair(Orthopair o){
		if(getFamily().isEmpty()){
			getFamily().add(o);
			return true;
		}else if(getFamily().get(0).getUniverse().equals(o.getUniverse())){
			if(!overlap){
				for(Orthopair p : getFamily()){
					Orthopair tmp1 = new Orthopair(o);
					tmp1.getP().addAll(tmp1.getBnd());
					Orthopair tmp2 = new Orthopair(o);
					if(tmp1.getP().retainAll(p.getP()) || tmp2.getP().retainAll(p.getBnd()))
						return false;
				}
				getFamily().add(o);
				return true;
			}else{
				getFamily().add(o);
				return true;
			}
		}
		return false;
	}
	
	/**
	 * Compute the total number of elements in the boundaries
	 * @return the number of elements in the boundaries
	 */
	public int totalBoundary(){
		return getFamily().stream().map(o -> o.getBnd()).reduce((b1, b2) -> {
			HashSet<Instance> t = new HashSet<Instance>(b1);
			t.addAll(b2);
			return t;
		}).get().size();
	}
	
	/**
	 * Compute the total number of elements in the boundaries for the given list
	 * @param l, a collection of orthopairs
	 * @return the number of elements in the boundaries
	 */
	public static int totalBoundary(ArrayList<Orthopair> l){
		return l.stream().map(o -> o.getBnd()).reduce((b1, b2) -> {
			HashSet<Instance> t = new HashSet<Instance>(b1);
			t.addAll(b2);
			return t;
		}).get().size();
	}
	
	/**
	 * Compute the value of the lower entropy of the orthopartition
	 * @return the lower entropy
	 * @throws Exception
	 */
	public double lowerEntropy() throws Exception{
		double lowerEntropy = 0;
		ArrayList<Orthopair> tmp = new ArrayList<Orthopair>();
		for(Orthopair o : getFamily()){
			tmp.add(new Orthopair(o));
		}
		if(!overlap){
			while(Orthopartition.totalBoundary(tmp) != 0){
				Orthopair max = null;
				for(Orthopair o : tmp){
					if(max == null && o.entropy() > 0)
						max = o;
					if(o.entropy() > 0 && o.getUpperSize() > max.getUpperSize())
						max = o;
				}
				max.getP().addAll(max.getBnd());
				max.setBnd(new HashSet<Instance>());
				for(Orthopair o : tmp){
					if(o != max){
						o.getBnd().removeAll(max.getP());
						o.getN().addAll(max.getP());
					}
				}
			}
			for(Orthopair o : tmp)
				for(Orthopair p : tmp)
					if(o != p)
						lowerEntropy += o.getLowerSize()*p.getLowerSize();
		} else{
			while(Orthopartition.totalBoundary(tmp) != 0){
				for(Orthopair o : tmp){
					o.getP().addAll(o.getBnd());
					o.setBnd(new HashSet<Instance>());
				}
			}
			for(Orthopair o : tmp)
				for(Orthopair p : tmp)
					if(o != p){
						Orthopair to = new Orthopair(o);
						Orthopair tp = new Orthopair(p);
						to.getP().removeAll(to.intersect(tp).getP());
						lowerEntropy += to.getLowerSize()*tp.getLowerSize();
					}
		}
		return lowerEntropy/(tmp.get(0).getUniverseSize()*tmp.get(0).getUniverseSize());
	}
	
	/**
	 * Compute the value of the upper entropy of the orthopartition
	 * @return the upper entropy
	 * @throws Exception
	 */
	public double upperEntropy() throws Exception{
		double upperEntropy = 0;
		ArrayList<Orthopair> tmp = new ArrayList<Orthopair>();
		for(Orthopair o : getFamily()){
			tmp.add(new Orthopair(o));
		}
			while(Orthopartition.totalBoundary(tmp) != 0){
				Orthopair min = null;
				for(Orthopair o : tmp){
					if(min == null && o.entropy() > 0)
						min = o;
					if(o.entropy() > 0 && o.getLowerSize() < min.getLowerSize())
						min = o;
				}
				Instance i = (Instance) min.getBnd().toArray()[0];
				min.getP().add(i);
				min.getBnd().remove(i);
				for(Orthopair o : tmp){
					if(o != min){
						o.getBnd().removeAll(min.getP());
						o.getN().addAll(min.getP());
					}
				}
			}
			for(Orthopair o : tmp)
				for(Orthopair p : tmp)
					if(o != p)
						if(!overlap)
							upperEntropy += o.getLowerSize()*p.getLowerSize();
						else{
							Orthopair to = new Orthopair(o);
							Orthopair tp = new Orthopair(p);
							to.getP().removeAll(to.intersect(tp).getP());
							upperEntropy += to.getLowerSize()*tp.getLowerSize();
						}
		return upperEntropy/(tmp.get(0).getUniverseSize()*tmp.get(0).getUniverseSize());
	}
	
	/**
	 * Compute the meet orthopartition
	 * @param pi, another orthopartition
	 * @return the meet of this and pi
	 * @throws Exception
	 */
	public Orthopartition meet(Orthopartition pi) throws Exception{
		ArrayList<Orthopair> tmp = new ArrayList<Orthopair>();
		for(Orthopair o : getFamily()){
			for(Orthopair p : pi.getFamily())
				if(!o.intersect(p).isEmpty())
					tmp.add(o.intersect(p));
		}
		boolean overlap = this.overlap || pi.overlap;
		return new Orthopartition(tmp, overlap);
	}
	
	/**
	 * Compute the mutual information between this orthopartition and the given one
	 * @param pi, another orthopartition
	 * @return the value of the mutual information
	 * @throws Exception
	 */
	public double mutualInformation(Orthopartition pi) throws Exception{
		double result = 0;
		double result1 = (this.lowerEntropy() + this.upperEntropy())/2;
		double result2 = (pi.lowerEntropy() + pi.upperEntropy())/2;
		Orthopartition m = this.meet(pi);
		result = result1 + result2 - (m.lowerEntropy() + m.upperEntropy())/2;
		if(result1 > result2)
			result /= result1;
		else
			result /= result2;
		return result;
	}
	
	/**
	 * Checks if the given instance is in boundary
	 * @param i, instance
	 * @return whether the given instance is in a boundary
	 */
	public boolean inBoundary(Instance i) {
		boolean result = false;
		for(Orthopair o : getFamily()) {
			if(o.getBnd().contains(i)) {
				result = true;
				break;
			}
		}
		return result;
	}
	
	/**
	 * Checks the number of boundaries for the given instance
	 * @param i, instance
	 * @return the number of boundaries for the given instance
	 */
	public int numBoundaries(Instance i){
		int result = 0;
		for(Orthopair o : getFamily())
			if(o.getBnd().contains(i))
				result++;
		return result;
	}
	
	
	/**
	 * Compute the purity between this orthopartition and the given one
	 * @param op, another orthopartition
	 * @return the value of the purity
	 * @throws Exception
	 */
	public double purity(Orthopartition op) throws Exception{
		double result = 0;
		for(Orthopair o : getFamily()){
			double iSize = 0;
			for(Orthopair p : op.getFamily()){
				double size = o.intersect(p).getLowerSize();
				for(Instance i : o.getBnd())
					if(p.getP().contains(i)){
						size += ((double) 1.0/numBoundaries(i));
					}
				if(size > iSize)
					iSize = size;
			}
			result += iSize;
		}
		return result/op.getFamily().get(0).getUniverseSize();
	}
	
	public String toString(){
		String result = "";
		for(Orthopair o : getFamily())
			result += o.toString() + "\n\n";
		return result;
	}
	
	/**
	 * Compute the orthopair containing the given instance
	 * @param x, an instance
	 * @return the collection of orthopairs containing the instance
	 */
	public ArrayList<Integer> inWhich(Instance x){
		ArrayList<Integer> r = new ArrayList<Integer>();
		for(int i = 0; i < getFamily().size(); i++)
			if(getFamily().get(i).getP().contains(x) || getFamily().get(i).getBnd().contains(x))
					r.add(i);
		return r;
	}

	public ArrayList<Orthopair> getFamily() {
		return family;
	}

	public void setFamily(ArrayList<Orthopair> family) {
		this.family = family;
	}
		
}

