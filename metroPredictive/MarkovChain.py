import numpy as np

class MarkovChain:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.failure_state = n_clusters   # failure state index
        self.n_states = n_clusters + 1   # clusters + failure

        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        self.counts = np.zeros((self.n_states, self.n_states)) 

        self.hit_sum = np.zeros(self.n_states)   # sum of times to failure
        self.hit_count = np.zeros(self.n_states) # number of observations


    def fit(self, sequences):

        for seq in sequences:
            seq = np.array(seq) 

            
            fail_indices = np.where(seq == self.failure_state)[0]
            fail_index = fail_indices[0] if len(fail_indices) > 0 else None

           
            for i, j in zip(seq[:-1], seq[1:]):
                self.transition_matrix[i, j] += 1

           
            if fail_index is not None:
                for t, state in enumerate(seq[:fail_index]):
                    self.hit_sum[state] += (fail_index - t)
                    self.hit_count[state] += 1

        self._normalize()

    def partial_fit(self, sequence): # won't work, need to fix to work with numpy arrays 

        time_to_fail = None
        if self.failure_state in sequence:
            fail_index = sequence.index(self.failure_state)
            time_to_fail = fail_index

        for (i, j) in zip(sequence[:-1], sequence[1:]):
            self.counts[i, j] += 1

        if time_to_fail is not None:
            for t, state in enumerate(sequence[:fail_index]):
                self.hit_sum[state] += (time_to_fail - t)
                self.hit_count[state] += 1

        self._normalize()

    def _normalize(self):
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            self.transition_matrix, row_sums, 
            where=row_sums != 0
        )

    def predict_next(self, current_state):
        return np.random.choice(self.n_states, p=self.transition_matrix[current_state])

    def expected_time_to_failure(self, current_state):
        if self.hit_count[current_state] == 0:
            return np.inf
        return self.hit_sum[current_state] / self.hit_count[current_state]

    def most_likely_path_to_failure(self, start_state, max_steps=50):
        path = [start_state]
        current = start_state
        steps = 0

        while current != self.failure_state and steps < max_steps:
            next_state = np.argmax(self.transition_matrix[current])
            path.append(next_state)
            current = next_state
            steps += 1

        return path
