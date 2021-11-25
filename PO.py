
import numpy as np
import matplotlib.pyplot as plt


def initialization(SearchAgents_no,dim,ub,lb):
    Boundary_no = np.shape(ub)[0]
    Positions = np.zeros([SearchAgents_no,dim])

    if Boundary_no>1:
        for i in range(dim):
            ub_i=ub[i]
            lb_i=lb[i]
            Positions[:,i]=np.random.random([SearchAgents_no,1])*(ub_i-lb_i)+lb_i
    if Boundary_no==1:
        Positions=np.random.random([SearchAgents_no,dim])*(ub-lb)+lb
    return Positions

def fitness_function(pop):
    return np.mean(pop**2)

def PO(SearchAgents_no,areas,parties,lamda,Max_iter,lb,ub,dim):
    Leader_pos = np.zeros([1, dim])
    Leader_score = np.inf
    Positions = initialization(SearchAgents_no, dim, ub, lb)
    auxPositions = Positions.copy()
    prevPositions = Positions.copy()
    Convergence_curve = np.zeros([Max_iter])
    fitness = np.zeros([SearchAgents_no, 1])

    ## % % % % % % % % % % % % % % % % Election Phase % % % % % % % % % % % %
    for i in range(np.shape(Positions)[0]):
        Flag4ub = Positions[i,:] > ub
        Flag4lb = Positions[i,:] < lb
        Positions[i,:]=(Positions[i,:]* (~(Flag4ub + Flag4lb)))+ub * Flag4ub + lb * Flag4lb
        fitness[i, 0] = fitness_function(Positions[i,:])
        if fitness[i, 0] < Leader_score: # # Change this to > for maximization problem
            Leader_score = fitness[i, 0]
            Leader_pos = Positions[i,:]


    auxFitness = fitness.copy()
    prevFitness = fitness.copy()

    #%%%%%%%%%%%%%%%%%%%%% Govt. Formation %%%%%%%%%%%%%%%%%%%%%%%%%%
    aWinnerInd=np.zeros([areas,1])   #Indices of area winners in x
    aWinners = np.zeros([areas,dim]) #Area winners are stored separately
    for a in range(areas):
        val=fitness[np.arange(a,SearchAgents_no,areas)]
        aWinnerFitness=np.min(val)
        aWinnerParty=np.argmin(val)
        aWinnerInd[a,0] = (aWinnerParty-1) * areas + a
        aWinnerInd=aWinnerInd.astype(np.int)
        aWinners[a,:] = Positions[aWinnerInd[a,0],:]
    # Finding party leaders
    pLeaderInd=np.zeros([parties,1])    #Indices of party leaders in x
    pLeaders = np.zeros([parties,dim])  #Positions of party leaders in x
    for p in range(parties):
        pStIndex = (p-1) * areas + 1
        pEndIndex = pStIndex + areas - 1
        leadIndex=np.argmin(fitness[np.arange(pStIndex,pEndIndex)])
        pLeaderInd[p,0] = (pStIndex - 1) + leadIndex #Indexof party leader
        pLeaderInd=pLeaderInd.astype(np.int)
        pLeaders[p,:] = Positions[pLeaderInd[p,0],:]


    t = 0 # Loop counter
    while t < Max_iter:
        prevFitness = auxFitness.copy()
        prevPositions = auxPositions.copy()
        auxFitness = fitness.copy()
        auxPositions = Positions.copy()

        # % % % % % % % % % % % % % % % % % % % % % Election campaign % % % % % % % % % % % % % % % % %
        for whichMethod in range(1,3):
            for a in range(areas):
                for p in range(parties):
                    i = (p - 1) * areas + a # index of member

                    for j in range(dim):
                        if whichMethod == 1: # position - updating w.r.t party leader
                            center = pLeaders[p, j]
                        elif whichMethod == 2: # position - updating    w.r.t    area    winner
                            center = aWinners[a, j]

                        # Cases of Eq. 9 in paper
                        if prevFitness[i] >= fitness[i]:
                            if ((prevPositions[i, j] <= Positions[i, j]) & (Positions[i, j] <= center)) | ((prevPositions[i, j] >= Positions[i, j]) & (Positions[i, j] >= center)):
                                radius = center - Positions[i, j]
                                Positions[i, j] = center + np.random.random() * radius
                            elif ((prevPositions[i, j] <= Positions[i, j]) & (Positions[i, j] >= center) & (center >= prevPositions[i, j])) | ((prevPositions[i, j] >= Positions[i, j]) & (Positions[i, j] <= center) & (center <= prevPositions[i, j])):
                                radius = abs(Positions[i, j] - center)
                                Positions[i, j] = center + (2 * np.random.random() - 1) * radius
                            elif ((prevPositions[i, j] <= Positions[i, j]) & (Positions[i, j] >= center) & (center <= prevPositions[i, j])) | ((prevPositions[i, j] >= Positions[i, j]) & (Positions[i, j] <= center) & (center >= prevPositions[i, j])):
                                radius = abs(prevPositions[i, j] - center)
                                Positions[i, j] = center + (2 * np.random.random() - 1) * radius

                        # Cases of Eq. 10 in paper
                        elif prevFitness[i]< fitness[i]:
                            if ((prevPositions[i, j] <= Positions[i, j]) & (Positions[i, j] <= center)) | ((prevPositions[i, j] >= Positions[i, j]) & (Positions[i, j] >= center)):
                                radius = abs(Positions[i, j] - center)
                                Positions[i, j] = center + (2 * np.random.random() - 1) * radius
                            elif ((prevPositions[i, j] <= Positions[i, j]) & (Positions[i, j] >= center) & (center >= prevPositions[i, j])) | ((prevPositions[i, j] >= Positions[i, j]) & (Positions[i, j] <= center) & (center <= prevPositions[i, j])):
                                radius = Positions[i, j] - prevPositions[i, j]
                                Positions[i, j] = prevPositions[i, j] + np.random.random() * radius
                            elif ((prevPositions[i, j] <= Positions[i, j]) & (Positions[i, j] >= center) & (center <= prevPositions[i, j])) | ((prevPositions[i, j] >= Positions[i, j]) & (Positions[i, j] <= center) & (center >= prevPositions[i, j])):
                                center2 = prevPositions[i, j]
                                radius = abs(center - center2)
                                Positions[i, j] = center + (2 * np.random.random() - 1) * radius

        #% % % % % % % % % % % % % % % % % % Party switching Phase % % % % % % % % % % % % % % % % %
        psr = (1 - t * ((1) / Max_iter)) * lamda
        for p in range(parties):
            for a in range(areas):
                fromPInd = (p - 1) * areas + a
                if np.random.random() < psr:
                    # Selecting a party other than current where want to send the member
                    toParty = np.random.randint(0,parties)
                    while (toParty == p):
                        toParty = np.random.randint(parties)

                    # Deciding member in TO party
                    toPStInd = (toParty - 1) * areas + 1
                    toPEndIndex = toPStInd + areas - 1
                    toPLeastFit = np.argmax(fitness[np.arange(toPStInd,toPEndIndex)])
                    toPInd = toPStInd + toPLeastFit - 1

                # Deciding what to do with member in FROM party and switching
                fromPInd = (p - 1) * areas + a
                temp = Positions[toPInd,:]
                Positions[toPInd,:] = Positions[fromPInd]
                Positions[fromPInd,:]=temp

                temp = fitness[toPInd]
                fitness[toPInd] = fitness[fromPInd]
                fitness[fromPInd] = temp

        ## % % % % % % % % % % % % % % % % Election Phase % % % % % % % % % % % %
        for i in range(np.shape(Positions)[0]):
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            fitness[i, 0] = fitness_function(Positions[i, :])
            if fitness[i, 0] < Leader_score:  # # Change this to > for maximization problem
                Leader_score = fitness[i, 0]
                Leader_pos = Positions[i, :]

        auxFitness = fitness.copy()
        prevFitness = fitness.copy()

        # %%%%%%%%%%%%%%%%%%%%% Govt. Formation %%%%%%%%%%%%%%%%%%%%%%%%%%
        aWinnerInd = np.zeros([areas, 1])  # Indices of area winners in x
        aWinners = np.zeros([areas, dim])  # Area winners are stored separately
        for a in range(areas):
            val=fitness[np.arange(a, SearchAgents_no, areas)]
            # aWinnerFitness = np.min(val)
            aWinnerParty=np.argmin(val)
            aWinnerInd[a, 0] = (aWinnerParty - 1) * areas + a
            aWinnerInd=aWinnerInd.astype(np.int)
            aWinners[a, :] = Positions[aWinnerInd[a, 0], :]
        # Finding party leaders
        pLeaderInd = np.zeros([parties, 1])  # Indices of party leaders in x
        pLeaders = np.zeros([parties, dim])  # Positions of party leaders in x
        for p in range(parties):
            pStIndex = (p - 1) * areas + 1
            pEndIndex = pStIndex + areas - 1
            leadIndex = np.argmin(fitness[np.arange(pStIndex, pEndIndex)])
            pLeaderInd[p, 0] = (pStIndex - 1) + leadIndex  # Indexof party leader
            pLeaderInd=pLeaderInd.astype(np.int)
            pLeaders[p, :] = Positions[pLeaderInd[p, 0], :]

        # % % % % % % % % % % % % % % % % % % % % % Parliamentarism % % % % % % % % % % % % % % % %
        for a in range(areas):
            newAWinner = aWinners[a,:]
            i = aWinnerInd[a]

            toa = np.random.randint(0,areas)
            while (toa == a):
                toa = np.random.randint(0,areas)

            toAWinner = aWinners[toa,:]
            for j in range(dim):
                distance = abs(toAWinner[j] - newAWinner[j])
                newAWinner[j] = toAWinner[j] + (2 * np.random.random() - 1) * distance

            newAWFitness = fitness_function(newAWinner[:])

            # Replace only if improves
            if newAWFitness < fitness[i]:
                Positions[i,:] = newAWinner[:]
                fitness[i] = newAWFitness
                aWinners[a,:] = newAWinner[:]

        Convergence_curve[t] = Leader_score
        t = t + 1
        print(t)

    plt.figure()
    plt.plot(Convergence_curve)
    plt.show()


if __name__ == '__main__':
    # % % % % % % % % % % % % % % % % Adjustable parameters % % % % % % % % % % % % % % % % %
    parties = 8 #Number of political parties
    lamda = 1.0 # Max limit of party switching rate

    areas = parties
    populationSize=parties * areas # Number of search agents
    Max_iteration = 100

    lb=np.array([0])
    ub=np.array([1])
    dim=10
    PO(populationSize,areas,parties,lamda,Max_iteration,lb,ub,dim)

