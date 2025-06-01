-**************************
Necessary functions
**************************-
loadPackage("NumericalAlgebraicGeometry")

-- r: depth of Picard iteration, l: length of multi-index I
Q = (r,l) -> ( 
    if (r==0 or l==0) then return 0;
    if (l==1) then return sub(2^(r-1),ZZ);
    return (sub(2^r-1,ZZ));
)



-- the list of multi-indices in {i,...,m} with length l
multiIndex = (l,i,m) -> (
    A := set splice {i..m};
    B := A^**l;
    for i from 1 to l-2 do B = B/splice;
    B = toList B;
    return for b in B list toList b
)




-- gives the position of word I in the list of multi-indices with elements in {i,...,m}
pos = (I, i, m) -> (
    p := length I;
    return sum(for l from 0 to p-1 list (m-i+1)^l) + position(multiIndex(p,i,m), el->el==I)
)




-- gives a list of matrices A_0, \ldots, A_R whose rows are indexed by the first NI words I and whose columns are indexed by words J. 
-- for all r = 0, \ldots, R, the (I,J)-entry of the matrix A_r is the polynomial \alpha_{r,J}^I.
-- R: maximum depth of Picard iterations, NI: number of words I , pars: parameters theta, t: endpoint of the time interval
polyAlpha = (q, m, n, R, NI, pars, t) -> (
    S := ring pars_1;
    -- Finds the maximum length of words I possible, saves it in L
    L := 0;
    s := 1;
    while s<NI do (L = L+1; s = s+(m+1)^L);
    L = max(q,L);
    NII := max(sum(for l from 0 to q list (m+1)^l),NI);

    
    -- r=0
    A := mutableMatrix map(S^NII, sum(for l from 0 to Q(0,L) list (n+1)^l), (i,j)->if i==0 and j==0 then 1 else 0);
    pList := {A}; -- Will eventually cotain R+1 matrices of size NII*Q(R,L)




    for r from 1 to R do (
        print r;
        A := mutableMatrix map(S^NII, sum(for l from 0 to Q(r,L) list (n+1)^l), (i,j)->0);
        
        -- alpha_{r, emptyset}^{emptyset}
        A_(0,0) = 1;

        
        -- Finding alpha_{r,J}^{(i)}
        for i from 0 to m do (
            for JLength from 0 to Q(r-1,q) do (
                for J in multiIndex(JLength, 0, n) do (
                    for j from 0 to n do (
                        s := 0;
                        for k in reverse(toList(0..q)) when JLength <= Q(r-1, k) do (
                            ML := multiIndex(k,0,m);
                            for K in ML do (
                                s = s + (pList_(r-1))_(pos(K,0,m),pos(J,0,n))*(pars_i)_(j,pos(K,0,m));
                            )
                        );
                        A_(pos({i},0,m), pos(append(J,j),0,n)) = s;
                    )
                )
            )
        );

        print "done with length 1";

        -- Finding alpha_{r,J}^I with |I|>=2
        for ILength from 2 to L do (
            print "ILength is"; print ILength;
            for JLength from 1 to Q(r-1, ILength-1) + Q(r,1) do (
                for I in multiIndex(ILength, 0, m) when pos(I,0,m)<NII do ( 
                    for J in multiIndex(JLength, 0, n) do (
                        if r == R and ENS(J,t)==0 then continue;
                        print "Jlength is"; print JLength;
                        Jnew := J_{0..JLength - 2};
                        s = 0;
                        for l from 0 to Q(r-1,ILength-1) do (
                            if length Jnew - l >= 0 and length Jnew - l <= Q(r,1) - 1 then (
                                for Lind in subsets(toList (0..length Jnew-1), l) do (
                                    L := Jnew_Lind;
                                    K := Jnew_(toList (0..length Jnew-1) - set Lind);
                                    s = s + (pList_(r-1))_(pos(I_{0.. length(I)-2},0,m), pos(L,0,n)) * A_(pos({last I},0,m), pos(append(K, last J),0,n));
                                )
                            )
                        );
                        A_(pos(I,0,m), pos(J,0,n)) = s;
                        -- print "pos(I,0,m) is"; print pos(I,0,m);
                        -- print pos(J,0,n);
                    )
                )
            )
        );


        -- print A;
        pList = append(pList, A)
    );
    return pList
)




-- parameters
-- returns a list of m+1 matrices of size (n+1)*(1+(m+1)+...+(m+1)^q) such that for each i \in [m]_0, A_i is the ith matrix with 
-- (A_i)_{j, I} = \theta_{i,j}^I for all j \in [n]_0 and all words I in [m]_0 of length at most q
params = (q,m,n) -> ( 
    N := sub(sum(for l from 0 to q list (m+1)^l),ZZ);
    par := map(RR^(n+1), N, (j,k)-> if j==0 and k==0 then 1 else 0);
    pars := {par};

    R := RR[th_(1,0,1)..th_(m,n,N)];

    for i from 1 to m do (
        pars = append(pars, map(R^(n+1), N, (j,k)->th_(i,j,k+1)))
    );

    return pars;
)


-- expected signature of the Brownian motion
-- gives the expected value of the Jth coordinate of the signature of (t,Brownian motion) at time t
ENS = (J,t) -> ( 
    nonzero := 0;
    flag = false;
    for i from 0 to length(J) - 1 do (
        if flag then flag = false
        else if J_i != 0 then (
            if i == length(J)-1 or J_(i+1) != J_i then (
                return 0
            )
            else (
                flag = true;
                nonzero = nonzero + 2
            )
        )
    );
    qJ := sub(nonzero/2 + (length(J) - nonzero),ZZ);
    pJ := sub((-nonzero/2),ZZ);
    return 2^pJ * t^qJ/(qJ!)
)


-- finds the point p in the list L with the minimum distance \sum_{i=0}^(length Pbase - 1) |w_i(p_i - Pbase_i)| from Pbase 
-- Pbase is a list, and L is a list of points whose length is the same as Pbase
-- w is a list of weights the same length as Pbase
minNorm = (Pbase, L, w) -> ( 
    if length L == 0 then (
        return "empty set"
    );
    a := coordinates L_0;
    d := sum(for i from 0 to length Pbase - 1 list w_i*abs(Pbase_i - a_i));
    -- print d;
    for P in L_{1..length L - 1} do (
        anew := coordinates P;
        dnew := sum(for i from 0 to length Pbase - 1 list w_i*abs(Pbase_i - anew_i));
        -- print dnew;
        if  dnew < d then (
            a = anew;
            d = dnew;
        )
    );
    return a;
)



-*********************************************************************************************************************************************************
Expected Signature Matching Method 
Input: m,n,q,qq,t,R,N,NSG,numTrials,knowns,knownsubs,setOfWords,Pbase,w,bigExpectedSignature (as explained in the config file)
Output: two list solutions1 and solutions2
        solutions1 contains the closest solution of the polynomial system to the true parameter (Pbase) with respect to the weight vector w in each trial
        solutions2 contains all the solutions of the polynomial system in each trial
*********************************************************************************************************************************************************-





ESMM = (m,n,q,qq,t,R,N,NSG,numTrials,knowns,knownsubs,setOfWords,Pbase,w,bigExpectedSignature) -> (

    -- gives the list of words in W([n]_0, 2^R-1)
    K = multiIndex(0,0,n);
    for i from 1 to 2^R-1 do (
        K = K | multiIndex(i,0,n);
    );


    -- computes the the N*1 vector y such that the rows of y are indexed by words I, and y_I is the polynomial P_r^I
    print "Computing the polynomials starts.";
    plist = polyAlpha(q,m,n,R,N,params(q,m,n),t);
    M = plist_R;
    S = ring M_(0,0);
    z = mutableMatrix map(S^(numcols M), 1, (i,j)-> ENS(K_i,t));
    y = mutableMatrix map(S^(numrows M), 1, (i,j) -> 0);
    for i from 0 to numrows M -1 do (
        s = 0;
        for j from 0 to numcols M - 1 do (
            -- print (i,j);
            if z_(j,0) != 0 then (
                s = s+M_(i,j)*z_(j,0);
            )
        );
        y_(i,0) = s;
    );



    -- creates a list of all parameters goodVars and a list of known values of parameters SUBS containing elemetns 
    -- of the form known_parameter => value_of_the_known_parameter
    goodVars = {};
    SUBS = {};
    for i from 0 to length knowns - 1 do (
        SUBS = SUBS | {sub(th_((knowns_i)_0,(knowns_i)_1,pos((knowns_i)_2,0,m)+1),S) => sub((knownsubs)_(i),S)};
        goodVars = goodVars | {sub(th_((knowns_i)_0,(knowns_i)_1,pos((knowns_i)_2,0,m)+1),S)}
    );

    for i from 1 to m do (
        for j from 0 to n do (
            for Ilength from 0 to q do (
                for I in multiIndex(Ilength, 0, m) do (
                    if not member((i,j,I),knowns) then (
                        -- SUBS = SUBS | {sub(th_(i,j,pos(I,0,m)+1),S) => 0};
                        goodVars = goodVars | {sub(th_(i,j,pos(I,0,m)+1),S)};
                    )
                    
                )
            )
        )
    );


    solutions1 = {};
    solutions2 = {}; 

    for bigCounter from 1 to numTrials do (
        print "bigCounter";
        print bigCounter;

        expectedSignature = bigExpectedSignature_{(bigCounter-1)*NSG..bigCounter*NSG-1};

        L = {}; -- list of polynomial equations
        listOfWords = {}; -- list of words corresponding to L

        K = multiIndex(0,0,m);
        for i from 1 to qq do ( 
            K = K | sort multiIndex(i,0,m);
        );

        -- Makes the list of polynomials in the polynomial system 
        for I in K do (
            i1 = pos(I, 0, m);
            i2 = position(K, el -> el== I);
            if i1<=N-1 and first degree y_(i1,0) > 0 then (
                L = L | {y_(i1,0) - expectedSignature_i2};
                listOfWords = listOfWords | {I};

            );
        );

        
        -- Makes the list of polynomials after substitutions corresponding to the known component of the parameters
        LL = {};
        -- counter = 0;
        LL = for P in L list sub(P, SUBS); -- Predetermines the values of the known parameters to make a zero-dimensional system
        X = for i from 0 to numcols(vars S)-1 list (vars S)_(0,i);
        Y = for i in goodVars list sub(i,S);
        SS = RR[X - (set Y_(toList (0..length knowns - 1)))]; -- Changes the ring accordingly
        LL2 = {};
        listOfWords2 = {};
        for P in LL do (if first degree P > 0 then (LL2 = LL2 | {P}; listOfWords2 = listOfWords2 | {listOfWords_(position(LL, el -> el==P))}));
        LL2 = for P in LL2 list sub(P, SS);

        
        -- print length LL2;
        -- lengthOfLL2s = lengthOfLL2s | {length LL2};


        
        if bigCounter == 1 then (
            SSS = {};
            for I in setOfWords do (
                if member(I, listOfWords2) then SSS = SSS | {position(listOfWords2, ell -> ell==I)}
                else return("bad set!")
            )
        );

        print "solving polynomial systems starts.";
        sup = set {};
        for P in LL2_SSS do (sup = sup + set support P);
        if length toList sup == length gens SS then (
            print "good set";
            T = solveSystem LL2_SSS;
            if length T > 0 then (
                print "non-empty solution set";
            );
            if w != splice {length setOfWords:0} then solutions1 = solutions1 | {{bigCounter, minNorm(Pbase, realPoints(T), w)}};
            
            for P in realPoints(T) do (
                solutions2 = solutions2 | {{bigCounter, P}}
            )
        )
        else return("bad set!");
    );


    return({solutions1, solutions2})

)




-*********************************************************************************************************************************************************
Experiments 6.2, 6.3, and 6.4 in the paper
To test on other configurations, prepare a similar config file.
*********************************************************************************************************************************************************-


load "config6.2.m2";
ESMM(m,n,q,qq,t,R,N,NSG,numTrials,knowns,knownsubs,setOfWords,Pbase,w,bigExpectedSignature)



load "config6.3.m2";
ESMM(m,n,q,qq,t,R,N,NSG,numTrials,knowns,knownsubs,setOfWords,Pbase,w,bigExpectedSignature)



load "config6.4.m2";
ESMM(m,n,q,qq,t,R,N,NSG,numTrials,knowns,knownsubs,setOfWords,Pbase,w,bigExpectedSignature)





























