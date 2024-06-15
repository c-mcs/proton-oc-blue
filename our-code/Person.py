import mesa
from extras import *
from itertools import chain
import networkx as nx

class Person(mesa.Agent):
    network_names = [
        'sibling',
        'offspring',
        'parent',
        'partner',
        'household',
        'friendship',
        'criminal']

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.neighbors = self.networks_init()
        self.num_crimes_committed = 0
        self.num_crimes_committed_this_tick = 0
        self.new_recruit = -2
        self.immigrant = False
        self.number_of_children = 0
        self.oc_member = False
        self.oc_role = None
        self.oc_boss = None
        self.oc_subordinates = None
        self.family_role = None
        self.arrest_weight = 0
        self.num_co_offenses = dict()  # criminal-links
        self.co_off_flag = dict()  # criminal-links
        self.job_level = 0
        self.wealth_level = 0
        self.criminal_tendency = 0
        self.mother = None
        self.father = None

    def init_person(self) -> None:
        """
        This method modifies the attributes of the person instance based on the model's
        stats_tables as part of the initial setup of the model agents.
        :return: None
        """

        row = weighted_one_of(self.model.age_gender_dist, lambda x: x[-1],
                                    self.model.random)  # select a row from our age_gender distribution
        self.birth_tick = 0 - row[0] * self.model.ticks_per_year
        self.ticks = 0
        self.age = row[0]
        self.gender_is_male = bool(row[1])  # ...and gender according to values in that row.
        self.retired = self.age >= self.model.retirement_age  # persons older than retirement_age are retired
        # education level is chosen, job and wealth follow in a conditioned sequence
        
        # Set education level based on age
        if self.age < 13:
            self.education_level = 1.0
        elif self.age < 18:
            self.education_level = min(2.0, pick_from_pair_list(self.model.edu[self.gender_is_male], self.model.random))
        elif self.age < 21:
            self.education_level = min(3.0, pick_from_pair_list(self.model.edu[self.gender_is_male], self.model.random))
        else:
            self.education_level = pick_from_pair_list(self.model.edu[self.gender_is_male], self.model.random)
            
        self.propensity = self.model.lognormal(self.model.nat_propensity_m, self.model.nat_propensity_sigma)



    def step(self):
        return None

    def networks_init(self):
        return {i: set() for i in Person.network_names}

    def make_friendship_link(self, asker):
        self.neighbors.get("friendship").add(asker)
        asker.neighbors.get("friendship").add(self)

    def make_partner_link(self, asker) -> None:
        """
        Create a two-way partner link in-place
        :param asker: Person
        :return: None
        """
        self.neighbors.get("partner").add(asker)
        asker.neighbors.get("partner").add(self)

    def make_parent_offsprings_link(self, asker):
        """
        Create a link between parent and offspring. Askers are the offspring.
        :param asker: Union[List[Person], Person]
        :return: None
        """
        if type(asker) == list:
            for person in asker:
                self.neighbors.get("offspring").add(person)
                person.neighbors.get("parent").add(self)
        else:
            self.neighbors.get("offspring").add(asker)
            asker.neighbors.get("parent").add(self)

    def add_sibling_link(self, targets):
        """
        Create a two-way sibling links in-place
        :param targets: List[Person]
        :return: None
        """
        for x in targets:
            if x != self:
                self.neighbors.get("sibling").add(x)
                x.neighbors.get("sibling").add(self)

    def make_household_link(self, targets):
        """
        Create a two-way household link in-place
        :param targets: List[Person]
        :return: None
        """
        for x in targets:
            if x != self:
                self.neighbors.get("household").add(x)
                x.neighbors.get("household").add(self)
    
    
    def add_criminal_link(self, asker):
        """
        Create a two-way criminal links in-place and update the connection weight
        :param asker: Person
        :return: None
        """
        self.neighbors.get("criminal").add(asker)
        self.num_co_offenses[asker] = 1
        asker.neighbors.get("criminal").add(self)
        asker.num_co_offenses[self] = 1

    def get_all_relatives(self):
        # Combine all the specified neighbor sets into a single set to avoid repetition
        unique_neighbors = set(self.neighbors['household']) | set(self.neighbors['sibling']) | \
                        set(self.neighbors['offspring']) | set(self.neighbors['parent']) | \
                        set(self.neighbors['partner'])
        return unique_neighbors
    
    def get_neighbor_list(self, net_name: str):
        """
        Given the name of a network, this method returns a list of agents within the network.
        If the network is empty, it returns an empty list.
        :param net_name: str, the network name
        :return: list, return an empty list if the network is empty
        """
        agent_net = self.neighbors.get(net_name)
        if len(agent_net) > 0:
            return list(agent_net)
        else:
            return []
    
    def get_all_agents(self):
        """
        Given a list of network names, this method returns a set of all agents
        within these networks, without repetitions.
        :param network_names: list, the list of network names
        :return: set, a set of all agents in all networks
        """
        all_agents = set()
        for net_name in self.network_names:
            agents = self.get_neighbor_list(net_name)
            all_agents.update(agents)
        return all_agents
    
    def remove_from_model_graphs(self):
        for family in self.model.families:
            if self in family:
                family.remove(self)
                break
        if self in self.model.friends_graph:
            self.model.friends_graph.remove_node(self)
        if self in self.model.crime_communities_graph:
            self.model.crime_communities_graph.remove_node(self)

    def remove_from_household(self) -> None:
        """
        This method removes the agent from household, keeping the networks consistent.
        Modify the Person.neighbors attribute in-place
        :return: None
        """
        for member in self.neighbors.get("household").copy():
            if self in member.neighbors.get("household"):
                member.neighbors.get("household").remove(self)
                self.neighbors.get("household").remove(member)
                
    def p_mortality(self):
        if self.age in self.model.mortality_table:
            p = self.model.mortality_table[self.age][self.gender_is_male] / self.model.ticks_per_year
        elif self.age > max(self.model.mortality_table):
            p = 1
        return p
    
    def die(self):
        neighbors = self.get_all_agents()
        for agent in neighbors:
            for network in agent.network_names:
                if self in agent.neighbors.get(network):
                    agent.neighbors.get(network).remove(self)
        self.model.schedule.remove(self)
        self.remove_from_model_graphs()

    def p_fertility(self) -> float:
        """
        Calculate the fertility
        :return: flot, the fertility
        """
        if np.min([self.number_of_children, 2]) in self.model.fertility_table[self.age]:
            return self.model.fertility_table[self.age][np.min([self.number_of_children, 2])] / self.model.ticks_per_year
        else:
            return 0
        
    def init_baby(self) -> None:
        """
        This method is for mothers only and allows to create new agents
        :return: None
        """
        self.number_of_children += 1
        self.model.number_born += 1
        index = len(self.model.agents) + 1
        
        new_agent = Person(index, self.model)
        self.model.schedule.add(new_agent)
        new_agent.age = 0
        new_agent.wealth_level = self.wealth_level
        new_agent.birth_tick = self.model.tick
        new_agent.wealth_level = self.wealth_level
        new_agent.mother = self
        
        if self.get_neighbor_list("offspring"):
            new_agent.add_sibling_link(self.get_neighbor_list("offspring"))
        self.make_parent_offsprings_link(new_agent)
        if self.get_neighbor_list("partner"):
            dad = self.get_neighbor_list("partner")[0]
            dad.make_parent_offsprings_link(new_agent)
            new_agent.father = dad
            new_agent.wealth_level = dad.wealth_level
        new_agent.make_household_link(self.get_neighbor_list("household"))

        new_agent.number_of_children = 0
        new_agent.oc_member = False
        new_agent.oc_role = None
        new_agent.oc_boss = None
        new_agent.oc_subordinates = None
        new_agent.family_role = "child"
        
        new_agent.ticks = 0

        new_agent.education_level = 1
        new_agent.gender_is_male = self.model.random.choice([True, False])  # True male False female

        new_agent.propensity = self.model.lognormal(self.model.nat_propensity_m, self.model.nat_propensity_sigma)
        new_agent.job_level = 0 # job e wealth level le mettiamo con un'altra funzione ma è bene ricordarci che sono variabili dell'agente
        new_agent.num_co_offenses = {}
        new_agent.networks_init()

    def break_friendship_link(self, asker):
        self.neighbors.get("friendship").remove(asker)
        asker.neighbors.get("friendship").remove(self)


    def get_caught(self) -> None:
        """
        When an agent is caught during a crime and goes to prison, this procedure is activated.
        :return: None
        """
        self.model.number_law_interventions_this_tick += 1
        self.model.people_jailed += 1
        self.prisoner = True
        if self.gender_is_male:
            self.sentence_countdown = pick_from_pair_list(self.model.male_punishment_length, self.model.random)
        else:
            self.sentence_countdown = pick_from_pair_list(self.model.female_punishment_length, self.model.random)
        self.sentence_countdown = self.sentence_countdown * self.model.punishment_length
        if self.job_level:
            self.job_level = 1
        # lose some friends?
        # we keep the friendship links and the family links

    def find_accomplices(self, n_of_accomplices: int):
        """
        This method is used to find accomplices during commit_crimes procedure
        :param n_of_accomplices: int, number of accomplices
        :return: List[Person]
        """
        if n_of_accomplices == 0:
            return [self]
        else:
            d = 1  # start with a network distance of 1
            accomplices = set()
            while len(accomplices) < n_of_accomplices and d <= self.model.max_accomplice_radius:
                # first create the group
                candidates = sorted(self.agents_in_radius(d), key=lambda x: self.candidates_weight(x))
                while len(accomplices) < n_of_accomplices and len(candidates) > 0:
                    candidate = candidates[0]
                    candidates.remove(candidate)
                    accomplices.add(candidate)
                    # todo: Should be if candidate.facilitator and facilitator_needed? tracked issue #234
                d += 1
            if len(accomplices) < n_of_accomplices:
                self.model.crime_size_fails += 1
            accomplices.add(self)
        return list(accomplices)
    
    
    def _agents_in_radius(self, context = network_names):
        """
        It finds the agents distant 1 in the specified networks, by default it finds it on all networks.
        :param context: List[str], limit to networks name
        :return: Set[Person]
        """
        agents_in_radius = set()
        for net in context:
            if self.neighbors.get(net):
                for agent in self.neighbors.get(net):
                    agents_in_radius.add(agent)
        return agents_in_radius

    def agents_in_radius(self, d: int, context = network_names):
        """
        It finds the agents distant "d" in the specified networks "context", by default it finds it on all networks.
        :param d: int, the distance
        :param context: List[str], limit to networks name
        :return: Set[Person]
        """
        # todo: This function must be speeded up, radius(3) on all agents with 1000 initial agents, t = 1.05 sec
        # todo: This function can be unified to neighbors_range
        radius = self._agents_in_radius(context)
        if d == 1:
            return radius
        else:
            for di in range(d - 1):
                for agent_in_radius in radius:
                    radius = radius.union(agent_in_radius._agents_in_radius(context))
            if self in radius:
                radius.remove(self)
            return radius
        
    def candidates_weight(self, agent) -> float:
        """
        This is what in the paper is called r - this is r R is then operationalised as the proportion
        of OC members among the social relations of each individual (comprising family, friendship, school,
        working and co-offending relations)
        :param agent: Person
        :return: float, the candidates weight
        """
        return -1 * (self.social_proximity(agent) * agent.oc_embeddedness() *
                     agent.criminal_tendency) if self.oc_member \
            else (self.social_proximity(agent) * agent.criminal_tendency)

    def social_proximity(self, target) -> int:
        """
        This function calculates the social proximity between self and another agent based on age,
        gender, wealth level, education level and friendship
        :param target: Person
        :return: int, social proximity
        """
        #todo: add weight? we could create a global model attribute(a dict) with weights
        total = 0
        total += 0 if abs(target.age - self.age) > 18 else 1 - abs(target.age - self.age)/18
        total += 1 if self.gender_is_male == target.gender_is_male else 0
        total += 1 if self.wealth_level == target.wealth_level else 0
        total += 1 if self.education_level == target.education_level else 0
        total += 1 if self.neighbors.get("friendship").intersection(
            target.neighbors.get("friendship")) else 0
        return total

    def n_links(self):
        result = 0
        for net in self.network_names:
            result += len(self.neighbors.get(net))
        return result
    
    def oc_embeddedness(self) -> float:
        """
        Calculates the cached_oc_embeddedness of self.
        :return: float, the cached_oc_embeddedness
        """
        if self.cached_oc_embeddedness is None:
            # only calculate oc-embeddedness if we don't have a cached value
            self.cached_oc_embeddedness = 0
            # start with an hypothesis of 0
            agents = self.agents_in_radius(self.model.oc_embeddedness_radius)
            oc_members = [agent for agent in agents if agent.oc_member]
            # this needs to include the caller
            agents.add(self)
            if oc_members:
                self.model.update_meta_links(agents)
                self.cached_oc_embeddedness = self.find_oc_weight_distance(oc_members) / self.find_oc_weight_distance(
                    agents)
        return self.cached_oc_embeddedness

    def find_oc_weight_distance(self, agents) -> float:
        """
        Based on the graph self.model.meta_graph calculates the weighted distance of self from each agent passed to the agents parameter
        :param agents: Union[Set[Person], List[Person]]
        :return: float, the distance
        """
        if self in agents:
            agents.remove(self)
        distance = 0
        for agent in agents:
            distance += 1 / nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.model.meta_graph,
                                                                                       self.unique_id, agent.unique_id,
                                                                                       weight='weight')
        return distance
    
    def update_criminal_tendency(self) -> None:
        """
        This procedure modifies the attribute self.criminal_tendency in-place, based on the individual characteristics of the agent.
        The original nomenclature of the model in Netlogo is: [employment, education, propensity, crim-hist, crim-fam, crim-neigh, oc-member]
        More information on criminal tendency modeling can be found on PROTON-Simulator-Report, page 30, 2.3.2 MODELLING CRIMINAL ACTIVITY (C):
        [https://www.projectproton.eu/wp-content/uploads/2019/10/D5.1-PROTON-Simulator-Report.pdf]
        :return: None
        """
        # employment
        self.criminal_tendency *= 1.30 if self.job_level == 1 else 1.0
        # education
        self.criminal_tendency *= 0.94 if self.education_level >= 2 else 1.0
        # propensity
        self.criminal_tendency *= 1.97 if self.propensity > (np.exp(
            self.model.nat_propensity_m - self.model.nat_propensity_sigma ** 2 / 2) + self.model.nat_propensity_threshold * np.sqrt(
            np.exp(self.model.nat_propensity_sigma) ** 2 - 1) * np.exp(
            self.model.nat_propensity_m + self.model.nat_propensity_sigma ** 2 / 2)) else 1.0
        # crim-hist
        self.criminal_tendency *= 1.62 if self.num_crimes_committed >= 0 else 1.0
        # crim-fam
        self.criminal_tendency *= 1.45 if self.family_link_neighbors() and (
                len([agent for agent in self.family_link_neighbors() if agent.num_crimes_committed > 0]) /
                len(self.family_link_neighbors())) > 0.5 else 1.0
        # crim-neigh
        friendship_neighbors = self.get_neighbor_list("friendship")

        # Calculate the proportion of friends who have committed crimes
        if friendship_neighbors:
            num_criminal_friends = len([agent for agent in friendship_neighbors if agent.num_crimes_committed > 0])
            proportion_criminal_friends = num_criminal_friends / len(friendship_neighbors)
        else:
            proportion_criminal_friends = 0

        # Update criminal tendency if more than 50% of friends have committed crimes
        self.criminal_tendency *= 1.81 if proportion_criminal_friends > 0.5 else 1.0

    def family_link_neighbors(self):
            """
            This function returns a list of all agents that have sibling,offspring,partner type connection with the agent.
            :return: List[Person], the agents
            """
            return self.get_neighbor_list("sibling") + self.get_neighbor_list("offspring") + self.get_neighbor_list(
                "partner")