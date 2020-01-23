package roughclustering;

import java.util.HashSet;
import java.util.stream.Collectors;

import weka.core.Instance;

/**
 * Implements an orthopair
 * @author Andrea Campagner
 *
 */
public class Orthopair {
	private HashSet<Instance> N;
	private HashSet<Instance> P;
	private HashSet<Instance> Bnd;
	
	public Orthopair(Orthopair o){
		P = new HashSet<Instance>(o.getP());
		Bnd = new HashSet<Instance>(o.getBnd());
		N = new HashSet<Instance>(o.getN());
	}
	
	public Orthopair(HashSet<Instance> n, HashSet<Instance> p, HashSet<Instance> bnd) throws Exception{
		super();
		N = n;
		HashSet<Instance> tmp = new HashSet<Instance>(p);
		tmp.retainAll(N);
		if(!tmp.isEmpty())
			throw new Exception("Sets are non-disjoint");
		P = p;
		setBnd(bnd);
	}
	
	public int getUniverseSize() {
		return P.size() + Bnd.size() + N.size();
	}
	
	public HashSet<Instance> getUniverse(){
		HashSet<Instance> result = new HashSet<Instance>();
		result.addAll(P);
		result.addAll(N);
		result.addAll(Bnd);
		return result;
	}

	public HashSet<Instance> getN() {
		return N;
	}

	public void setN(HashSet<Instance> n) throws Exception {
		HashSet<Instance> tmp1 = new HashSet<Instance>(n);
		tmp1.retainAll(P);
		HashSet<Instance> tmp2 = new HashSet<Instance>(n);
		tmp2.retainAll(Bnd);
		if(!tmp1.isEmpty() || !tmp2.isEmpty())
			throw new Exception("The sets are non-disjoint");
		N = n;
	}

	public HashSet<Instance> getP() {
		return P;
	}

	public void setP(HashSet<Instance> p) throws Exception{
		HashSet<Instance> tmp1 = new HashSet<Instance>(p);
		tmp1.retainAll(N);
		HashSet<Instance> tmp2 = new HashSet<Instance>(p);
		tmp2.retainAll(Bnd);
		if(!tmp1.isEmpty() || !tmp2.isEmpty())
			throw new Exception("The sets are non-disjoint");
		P = p;
	}

	public HashSet<Instance> getBnd() {
		return Bnd;
	}

	public void setBnd(HashSet<Instance> bnd) throws Exception{
		HashSet<Instance> tmp1 = new HashSet<Instance>(bnd);
		tmp1.retainAll(P);
		HashSet<Instance> tmp2 = new HashSet<Instance>(bnd);
		tmp2.retainAll(N);
		if(!tmp1.isEmpty() || !tmp2.isEmpty())
			throw new Exception("The sets are non-disjoint");
		Bnd = bnd;
	}
	
	/**
	 * Performs the join operation of the "truth" ordering
	 * @param o, an orthopair
	 * @return the join orthopair
	 * @throws Exception - the orthopairs are defined on different universes
	 */
	public Orthopair union(Orthopair o) throws Exception{
		if(!o.getUniverse().equals(getUniverse()))
			throw new Exception("Different universes");
		
		Orthopair result = new Orthopair(this);
		Orthopair tmp = new Orthopair(o);
		
		result.getN().retainAll(o.getN());
		
		result.getP().addAll(tmp.getP());
		
		result.getBnd().removeAll(result.getN());
		tmp.getBnd().removeAll(result.getN());
		result.getBnd().addAll(tmp.getBnd());
		result.getBnd().removeAll(result.getP());
		return result;
	}
	
	/**
	 * Performs the meet operation of the "truth" ordering
	 * @param o, an orthopair
	 * @return the meet orthopair
	 * @throws Exception - the orthopairs are defined on different universes
	 */
	public Orthopair intersect(Orthopair o) throws Exception{
		if(!o.getUniverse().equals(getUniverse()))
			throw new Exception("Different universes");
		
		Orthopair result = new Orthopair(this);
		Orthopair tmp = new Orthopair(o);
		
		result.getP().retainAll(o.getP());
		
		result.getN().addAll(tmp.getN());
		
		result.getBnd().removeAll(result.getP());
		tmp.getBnd().removeAll(result.getP());
		result.getBnd().addAll(tmp.getBnd());
		result.getBnd().removeAll(result.getN());
		return result;
	}
	
	/**
	 * Compute the boundary-based uncertainty measure
	 * @return the value of the boundary-based measure
	 */
	public double entropy(){
		return ((double) Bnd.size())/this.getUniverseSize();
	}
	
	/**
	 * Compute the size of the lower region (i.e. P)
	 * @return the size of the lower region
	 */
	public int getLowerSize(){
		return P.size();
	}
	
	/**
	 * Checks if the orthopair is empty (e.g. N == U)
	 * @return whether the orthopair is empty
	 */
	public boolean isEmpty(){
		return N.equals(getUniverse());
	}
	
	/**
	 * Compute the size of the upper region (i.e. P union Bnd)
	 * @return the size of the upper region
	 */
	public int getUpperSize(){
		return P.size() + Bnd.size();
	}
	
	public String toString(){
		return P.toString() + "\n" + Bnd.toString() + "\n" + N.toString();
	}
	
	public String prettyPrint(int att){
		String out = "";
		out += P.stream().map((Instance i) -> i.value(att)).collect(Collectors.toList()).toString() + "\n";
		out += Bnd.stream().map((Instance i) -> i.value(att)).collect(Collectors.toList()).toString() + "\n";
		out += N.stream().map((Instance i) -> i.value(att)).collect(Collectors.toList()).toString();
		return out;
	}
	
	public boolean equals(Orthopair o){
		return P.equals(o.getP()) && Bnd.equals(o.getBnd()) && N.equals(o.getN());
	}
}
