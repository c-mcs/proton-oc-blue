import os
import numpy as np
import mesa
import networkx as nx
from protonoc.simulator.extra import list_contains_problems
from Person import Person
from extras import *
import random
from itertools import combinations

class CrimeModel(mesa.Model):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    def __init__(self, N, model_params = {"no-params":"empty"}, agent_params = {"no-params":"empty"}):
        super().__init__()
        self.families = list()
        self.mafia_heads = set()
        self.ticks_per_year = 12
        self.nat_propensity_m: float = 1.0  # 0.1 -> 10
        self.nat_propensity_sigma: float = 0.25  # 0.1 -> 10.0
        self.education_modifier = 1.0
        self.parameters = model_params
        self.agent_params = agent_params
        self.initial_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.random = np.random.default_rng(1)
        self.random_state = random.randint(1,100)
        self.retirement_age = 64
        self.friends_graph = nx.Graph()
        self.crime_communities_graph = nx.Graph()
        self.number_arrests_per_year: int = 30  # 0 -> 100
        self.number_crimes_yearly_per10k: int = 2000 
        self.arrest_rate = self.number_arrests_per_year / self.ticks_per_year / self.number_crimes_yearly_per10k / 10000 * self.initial_agents
        self.num_oc_persons: int = 30  # 2 -> 200
        self.num_oc_families: int = 8  # 1 -> 50

        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Steps": lambda m: m.schedule.steps,
            }
        )
        self.setup()
    def setup(self):
        self.init_data_employed()
        self.init_agents()
        self.assign_jobs_and_wealth()
        self.generate_households()
        self.cleanup_unfit_individuals()
        self.setup_siblings()
        self.generate_friends_network()
        self.calculate_crime_multiplier()
        self.calculate_criminal_tendency()
        self.setup_oc_groups2()
        
    def init_data_employed(self):
        self.age_gender_dist = read_csv_data("initial_age_gender_dist").values.tolist()
        self.head_age_dist = df_to_dict(read_csv_data("head_age_dist_by_household_size"))
        self.proportion_of_male_singles_by_age = df_to_dict(read_csv_data("proportion_of_male_singles_by_age"))
        self.hh_type_dist = df_to_dict(read_csv_data("household_type_dist_by_age"))
        self.partner_age_dist = df_to_dict(read_csv_data("partner_age_dist"))
        self.children_age_dist = df_to_dict(read_csv_data("children_age_dist"))
        self.p_single_father = read_csv_data("proportion_single_fathers")
        self.edu = df_to_dict(read_csv_data("edu"))
        self.work_status_by_edu_lvl = df_to_dict(read_csv_data("work_status_by_edu_lvl"))
        self.wealth_quintile_by_work_status = df_to_dict(read_csv_data("wealth_quintile_by_work_status"))
        self.c_range_by_age_and_sex = df_to_lists(read_csv_data("crime_rate_by_gender_and_age_range"))
    
    def cleanup_unfit_individuals(self):
        for a in self.schedule.agents:
            if a.family_role == None:
                self.schedule.agents.remove(a)

    def household_sizes(self, size: int):
        """
        Loads a table with a probability distribution of household size and calculates household
        based on initial agents
        :param size: int, population size
        :return: list, the sizes of household
        """
        hh_size_dist = read_csv_data("household_size_dist").values
        sizes = []
        current_sum = 0
        while current_sum < size:
            hh_size = pick_from_pair_list(hh_size_dist, self.random)
            if current_sum + hh_size <= size:
                sizes.append(hh_size)
                current_sum += hh_size
        sizes.sort(reverse=True)
        return sizes

    def lognormal(self, mu, sigma) -> float:
        return np.exp(mu + sigma * self.random.normal())
    
    def init_agents(self):
        for i in range(self.initial_agents):
            new_agent = Person(i, self)
            new_agent.init_person()
            self.schedule.add(new_agent)

    def calculate_similarity(self, agent, potential_friend):
        age_diff = abs(agent.age - potential_friend.age)
        gender_diff = int(agent.gender_is_male != potential_friend.gender_is_male)
        education_diff = abs(agent.education_level - potential_friend.education_level)
        wealth_diff = abs(agent.wealth_level - potential_friend.wealth_level)

        age_weight = 3
        gender_weight = 2.0
        education_weight = 1.0
        wealth_weight = 1.0

        # Check if either the agent or the potential friend is a minor and if the age difference exceeds 4 years
        if (agent.age < 18 or potential_friend.age < 18) and age_diff > 4:
            similarity_score = 100.0  # Set a high similarity score to indicate low compatibility
        else:
            similarity_score = (age_weight * age_diff + 
                                gender_weight * gender_diff + 
                                education_weight * education_diff + 
                                wealth_weight * wealth_diff)
        return similarity_score

    def generate_friends_network(self):
        agents = list(self.schedule.agents)
        self.random.shuffle(agents) 
        
        # Calculate mean friend count based on age of agents
        typical_number_of_friends = 3
        mean_friend_counts = [typical_number_of_friends - (agent.age / 99) * 1.5 for agent in agents]  # Adjust this scaling factor as needed
        # Generate friend counts for each agent using a normal distribution with scaled mean
        friend_counts = [self.random.normal(loc=mean, scale=2) for mean in mean_friend_counts]
        friend_counts = np.clip(friend_counts, 0, typical_number_of_friends*2).astype(int)  # Ensure friend counts are between 0 and 6
        sample_size = min(int(self.initial_agents / 3), len(agents) - 1)  # Limit the sample size to 1/3(N) or total agents - 1
        for agent, num_friends in zip(agents, friend_counts):
            excluded_neighbors = agent.get_all_relatives()

            potential_friends = [a for a in agents if a != agent and a not in excluded_neighbors]
            if len(potential_friends) > sample_size:
                potential_friends = self.random.choice(potential_friends, sample_size, replace=False)
            
            if len(potential_friends) == 0:
                continue
            # Filter potential friends based on age difference exceeding 4 years
            potential_friends = [friend for friend in potential_friends if self.calculate_similarity(agent, friend) < 100.0]
            if len(potential_friends) == 0:
                continue
            
            similarity_scores = np.array([self.calculate_similarity(agent, a) for a in potential_friends])
            inv_similarity_scores = 1 / (similarity_scores + 1e-6)
            probabilities = inv_similarity_scores / inv_similarity_scores.sum()
            actual_num_friends = min(num_friends, len(potential_friends))  # Ensure actual_num_friends does not exceed potential friends
            
            if actual_num_friends != 0:
                r = random.random()
                if agent.gender_is_male and r < 0.20:
                    continue
                if not agent.gender_is_male and r < 0.05:
                    continue

            # Adjust number of friends based on gender bias with probability
            if agent.gender_is_male and actual_num_friends != 0 and self.random.random() < 0.8:
                actual_num_friends -= 1
                
            selected_friends = self.random.choice(potential_friends, size=actual_num_friends, replace=False, p=probabilities)
            for friend in selected_friends:
                agent.make_friendship_link(friend)
                
        # Create the friends graph
        self.friends_graph = nx.Graph()
        for agent in self.schedule.agents:
            self.friends_graph.add_node(agent.unique_id)
            for friend in agent.neighbors['friendship']:
                self.friends_graph.add_edge(agent.unique_id, friend.unique_id)

        # Debug: Print the number of edges to verify links
        print(f"Number of friendships: {self.friends_graph.number_of_edges()}")



    def generate_households(self) -> None:
            """
            This procedure aggregates eligible agents into households based on the tables
            (ProtonOC.self.head_age_dist, ProtonOC.proportion_of_male_singles_by_age,
            ProtonOC.hh_type_dist, ProtonOC.partner_age_dist, ProtonOC.children_age_dist,
            ProtonOC.p_single_father) and mostly follows the third algorithm from Gargiulo et al. 2010
            (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0008828)

            :return: None
            """
            population = self.schedule.agents
            self.hh_size = self.household_sizes(self.initial_agents)
            complex_hh_sizes = list()
            max_attempts_by_size = 50
            attempts_list = list()

            for size in self.hh_size:
                success = False
                nb_attempts = 0
                while not success and nb_attempts < max_attempts_by_size:
                    hh_members = list()
                    nb_attempts += 1
                    head_age = pick_from_pair_list(self.head_age_dist[size], self.random)
                    if size == 1:
                        male_wanted = (self.random.random()
                                    < self.proportion_of_male_singles_by_age[head_age])
                        head = pick_from_population_pool_by_age_and_gender(head_age,
                                                                            male_wanted,
                                                                            population, self.random)
                        if head:
                            success = True
                            attempts_list.append(nb_attempts)
                            head.family_role = "head"
                    else:
                        hh_type = pick_from_pair_list(self.hh_type_dist[head_age], self.random)
                        if hh_type == "single_parent":
                            male_head = self.random.random() \
                                        < float(self.p_single_father.columns.to_list()[0])
                        else:
                            male_head = True
                        if male_head:
                            mother_age = pick_from_pair_list(self.partner_age_dist[head_age],
                                                                self.random)
                        else:
                            mother_age = head_age
                        head = pick_from_population_pool_by_age_and_gender(head_age,\
                                                                        male_head,
                                                                        population,self.random)
                        if head:
                            hh_members.append(head)
                            head.family_role = "head"
                            if hh_type == "couple":
                                mother = pick_from_population_pool_by_age_and_gender(mother_age,\
                                                                                    False,
                                                                                    population,self.random)
                                if mother:
                                    hh_members.append(mother)
                                    mother.family_role = "partner"
                            num_children = size - len(hh_members)
                            for child in range(1, int(num_children) + 1):
                                if num_children in self.children_age_dist:
                                    if mother_age in self.children_age_dist[num_children]:
                                        child_age = pick_from_pair_list(
                                            self.children_age_dist[num_children][mother_age], self.random)
                                        child = pick_from_population_pool_by_age(child_age,
                                                                                population,self.random)
                                        if child:
                                            hh_members.append(child)
                                            child.family_role = "child"
                    hh_members = [memb for memb in hh_members if memb is not None]  # exclude Nones
                    if len(hh_members) == size:
                        success = True
                        attempts_list.append(nb_attempts)
                        family_wealth_level = hh_members[0].wealth_level
                        if hh_type == "couple":
                            hh_members[0].make_partner_link(hh_members[1])
                            couple = hh_members[0:2]
                            offsprings = hh_members[2:]
                            for partner in couple:
                                partner.make_parent_offsprings_link(offsprings)
                            for sibling in offsprings:
                                sibling.add_sibling_link(offsprings)
                        for member in hh_members:
                            member.make_household_link(hh_members)
                            member.wealth_level = family_wealth_level
                        self.families.append(hh_members)
                    else:
                        for member in hh_members:
                            population.add(member)
                if not success:
                    complex_hh_sizes.append(size)
            print(f"Removing {len(population)} agents.")
            for m in population:
                self.schedule.agents.remove(m)
            return
            for comp_hh_size in complex_hh_sizes:
                comp_hh_size = int(min(comp_hh_size, len(population)))
                complex_hh_members = population[0:comp_hh_size]  
                max_age_index = [x.age for x in complex_hh_members].index(max([x.age for x in complex_hh_members]))
                family_wealth_level = complex_hh_members[max_age_index].wealth_level
                if len(complex_hh_members) == 1:
                    population.remove(member)
                    member.family_role = "head"
                    self.families.append(member)
                for member in complex_hh_members:
                    population.remove(member)  
                    member.make_household_link(complex_hh_members)  
                    member.wealth_level = family_wealth_level
                    member.family_role = "complex"
                if len(complex_hh_members) > 1:
                    self.families.append(complex_hh_members)

    def setup_siblings(self) -> None:
        """
        Right now, during setup, links between agents are only those within households, between
        friends and related to the school. At this stage of the standard setup, agents are linked
        through "siblings" links outside the household. To simulate agents who have left the
        original household, agents who have children are taken and "sibling" links are created
        taking care not to create incestuous relationships.

        :return: None
        """
        agent_left_household = [p for p in self.schedule.agents if
                                p.neighbors.get('offspring')]
        # simulates people who left the original household.
        for agent in agent_left_household:
            num_siblings = self.random.poisson(0.5)
            # 0.5 -> the number of links is N^3 agents, so let's keep this low at this stage links
            # with other persons are only relatives inside households and friends.
            candidates = [c for c in agent_left_household
                          if c not in agent.neighbors.get("household")
                          and abs(agent.age - c.age) < 5 and c != agent]
            # remove couples from candidates and their neighborhoods (siblings)
            if len(candidates) >= 50:
                candidates = self.random.choice(candidates, 50, replace=False).tolist()
            while len(candidates) > 0 and list_contains_problems(agent, candidates):
                # trouble should exist, or check-all-siblings would fail
                potential_trouble = [x for x in candidates if agent.get_neighbor_list("partner")]
                trouble = self.random.choice(potential_trouble)
                candidates.remove(trouble)
            targets = [agent] + self.random.choice(candidates,
                                                   min(len(candidates), num_siblings)).tolist()
            for sib in targets:
                if sib in agent_left_household:
                    agent_left_household.remove(sib)
            for target in targets:
                target.add_sibling_link(targets)
                # this is a good place to remind that the number of links in the sibling link
                # neighbors is not the "number of brothers and sisters"
                # because, for example, 4 brothers = 6 links.
            other_targets = targets + [s for c in targets for s in c.neighbors.get('sibling')]
            for target in other_targets:
                target.add_sibling_link(other_targets)

    def assign_jobs_and_wealth(self) -> None:
        """
        This procedure modifies the job_level and wealth_level attributes of agents in-place.
        This is just a first assignment, and will be modified first by the multiplier then
        by adding neet status.

        :return: None
        """
        permuted_set = self.random.permuted(self.schedule.agents)
        for agent in permuted_set:
            if agent.age > 16:
                agent.job_level = pick_from_pair_list(
                    self.work_status_by_edu_lvl[agent.education_level][agent.gender_is_male],
                    self.random)
                agent.wealth_level = pick_from_pair_list(
                    self.wealth_quintile_by_work_status[agent.job_level][agent.gender_is_male],
                    self.random)
            else:
                agent.job_level = 1.0
                agent.wealth_level = 1.0

    def calculate_crime_multiplier(self) -> None:
            """
            Based on ProtonOC.c_range_by_age_and_sex this procedure modifies in-place
            the attribute ProtonOc.crime_multiplier
            :return: None
            """
            total_crime = 0
            for line in self.c_range_by_age_and_sex:
                people_in_cell = [agent for agent in self.schedule.agents if line[0][1] <
                                agent.age <= line[1][0] and agent.gender_is_male == line[0][0]]
                n_of_crimes = line[1][1] * len(people_in_cell)
                total_crime += n_of_crimes
            self.crime_multiplier = \
                self.number_crimes_yearly_per10k / 10000 * len(self.schedule.agents) / total_crime

    def calculate_criminal_tendency(self) -> None:
        """
        Based on the ProtonOC.c_range_by_age_and_sex distribution, this function calculates and
        assigns to each agent a value representing the criminal tendency. It modifies the attribute
        Person.criminal_tendency in-place.

        :return: None
        """
        for line in self.c_range_by_age_and_sex:
            # the line variable is composed as follows:
            # [[bool(gender_is_male), int(minimum age range)],
            # [int(maximum age range), float(c value)]]
            subpop = [agent for agent in self.schedule.agents if
                      line[0][1] <= agent.age <= line[1][0] and agent.gender_is_male == line[0][0]]
            if subpop:
                c = line[1][1]
                # c is the cell value. Now we calculate criminal-tendency with the factors.
                for agent in subpop:
                    agent.criminal_tendency = c
                    #agent.update_criminal_tendency() ! TODO
                # then derive the correction epsilon by solving
                # $\sum_{i} ( c f_i + \epsilon ) = \sum_i c$
                epsilon = c - np.mean([agent.criminal_tendency for agent in subpop])
                for agent in subpop:
                    agent.criminal_tendency += epsilon
        #if self.intervention_is_on() and self.facilitator_repression:
            #self.calc_correction_for_non_facilitators() # Possibilità di inserire qui una policy
    
    
    def setup_oc_groups(self) -> None:
        """
        This procedure creates "criminal" type links within the families, based on the criminal
        tendency of the agents, in case the agents within the families are not enough, new members
        are taken outside.
        :return: None
        """
        # OC members are scaled down if we don't have 10K agents
        scaled_num_oc_families = np.ceil(
            self.num_oc_families * self.initial_agents / 10000 * self.num_oc_persons / 30)
        scaled_num_oc_persons = np.ceil(
            self.num_oc_persons * self.initial_agents / 10000)
        members_per_family = int(scaled_num_oc_persons / scaled_num_oc_families)
        # families first.
        # we assume here that we'll never get a negative criminal tendency.
        oc_family_heads = weighted_n_of(scaled_num_oc_families, self.schedule.agents,
                                              lambda x: x.criminal_tendency, self.random)
        candidates = list()
        cosca = set()
        graphs = []
        for head in oc_family_heads:
            head.oc_member = True
            head.oc_role = "boss"
            self.mafia_heads.add(head)
            candidates += [relative for relative in head.get_all_relatives() if
                           relative.age >= 18]
            if len(candidates) >= members_per_family:  # family members will be enough
                members_in_families = weighted_n_of(members_per_family,
                                                        candidates,
                                                        lambda x: x.criminal_tendency,
                                                        self.random)
                # fill up the families as much as possible
                for member in members_in_families:
                    member.oc_member = True
                    member.oc_role = "soldier"
                    cosca.add(member)
            else:
                while members_per_family - len(cosca) > 0:
                    for candidate in candidates:
                        candidate.oc_member = True
                        candidate.oc_role = "soldier"
                        cosca.add(candidate)
                    break
                if members_per_family - len(cosca) > 0:
                    missing_n = members_per_family - len(cosca) 
                    for el in cosca:
                        out_of_family_candidates += [agent for agent in el.get_all_relatives()
                                                        if not agent.oc_member and agent.age >= 18]
                    out_of_family_candidates = weighted_n_of(
                        missing_n,
                        out_of_family_candidates, lambda x: x.criminal_tendency, self.random)
                    for out_of_family_candidate in out_of_family_candidates:
                        out_of_family_candidate.oc_member = True
                        out_of_family_candidate.oc_role = "soldier"
                        cosca.add(candidate)
            head.oc_subordinates = cosca
            cosca_graph = nx.Graph()
            cosca_graph.add_node(head)
            for cosca_member in cosca:
                cosca_graph.add_node(cosca_member)
                cosca_graph.add_edge(head, cosca_member)
            graphs.append(cosca_graph)
        # creiamo i nodi del grafo
        self.crime_communities_graph = nx.compose_all(graphs)
        for node1, node2 in combinations(self.mafia_heads, 2):
            self.crime_communities_graph.add_edge(node1, node2)
    
    def setup_oc_groups2(self) -> None:
        """
        This procedure creates "criminal" type links within the families, based on the criminal
        tendency of the agents. In case the agents within the families are not enough, new members
        are taken from outside, ensuring that members of the same household belong to the same mafia family.
        :return: None
        """
        # OC members are scaled down if we don't have 10K agents
        scaled_num_oc_families = int(np.ceil(
            self.num_oc_families * self.initial_agents / 10000 * self.num_oc_persons / 30))
        scaled_num_oc_persons = int(np.ceil(
            self.num_oc_persons * self.initial_agents / 10000))
        members_per_family = max(1, int(scaled_num_oc_persons / scaled_num_oc_families))  # Ensure at least 1 member per family
        print(f"Members per family: {members_per_family}")
        print(f"Families: {scaled_num_oc_families}")
        oc_family_heads = weighted_n_of(scaled_num_oc_families, self.schedule.agents,
                                        lambda x: x.criminal_tendency, self.random)
        graphs = []
        for head in oc_family_heads:
            head.oc_member = True
            head.oc_role = "boss"
            self.mafia_heads.add(head)
            
            # Collect family members ensuring household constraint
            candidates = self.collect_candidates(head)
            
            # Fill up the family with internal candidates first
            cosca = self.assign_family_members(head, candidates, members_per_family)
            
            # If not enough members, get additional members from outside
            if len(cosca) < members_per_family:
                missing_n = members_per_family - len(cosca)
                broader_family_candidates = self.collect_out_of_family_candidates(head, cosca, missing_n)
                additional_members = weighted_n_of(missing_n, broader_family_candidates, lambda x: x.criminal_tendency, self.random)
                cosca.update(self.assign_additional_members(additional_members, cosca, head))
            head.oc_subordinates = cosca
            cosca_graph = nx.Graph()
            cosca_graph.add_node(head)
            for cosca_member in cosca:
                cosca_graph.add_node(cosca_member)
                cosca_graph.add_edge(head, cosca_member)
            graphs.append(cosca_graph)

        self.crime_communities_graph = nx.compose_all(graphs)
        for node1, node2 in combinations(self.mafia_heads, 2):
            self.crime_communities_graph.add_edge(node1, node2)

    def collect_candidates(self, head):
        candidates = set()
        for relative in head.get_all_relatives():
            if relative.age >= 18:
                candidates.add(relative)
            candidates.update(m for m in relative.neighbors["household"] if m.age >= 18 and not m.oc_member)
        return candidates

    def assign_family_members(self, head, candidates, members_per_family):
        cosca = set()
        members_in_families = weighted_n_of(min(members_per_family, len(candidates)), candidates, lambda x: x.criminal_tendency, self.random)
        for member in members_in_families:
            member.oc_member = True
            member.oc_role = "soldier"
            member.oc_boss = head
            cosca.add(member)
        return cosca

    def collect_out_of_family_candidates(self, head, cosca, missing_n):
        out_of_family_candidates = set()
        for member in cosca:
            for agent in member.get_all_relatives():
                if any(a.oc_boss != head for a in agent.neighbors["household"]):
                    continue
                if agent.age >= 18 and agent.gender_is_male:
                    out_of_family_candidates.add(agent)
                    if len(out_of_family_candidates) >= missing_n:
                        return out_of_family_candidates
        return out_of_family_candidates

    def assign_additional_members(self, additional_members, cosca, head):
        for candidate in additional_members:
            candidate.oc_member = True
            candidate.oc_role = "soldier"
            candidate.oc_boss = head
            cosca.add(candidate)
            for household_member in candidate.neighbors["household"]:
                if household_member.age >= 18:
                    household_member.oc_member = True
                    household_member.oc_boss = head
        return cosca



