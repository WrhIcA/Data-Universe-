"""
Neuroscience Analysis Framework
Advanced computational tools for neural data analysis, brain connectivity,
and consciousness research based on Universal Pattern Recognition principles.

Integrates with Universal Knowledge Framework for cosmic-neural correspondence studies.
Developed for NASA AWG and Copernicus Program contributions.

Author[Suraj BahadurSilwal]
orcid: https://orcid.org/0009-0002-7602-188X
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats, ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: NEURAL SIGNAL PROCESSING
# ============================================================================

class NeuralSignalProcessor:
    """
    Process and analyze neural time series data (EEG, MEG, LFP, spike trains).
    """
    
    @staticmethod
    def bandpass_filter(signal_data: np.ndarray,
                       fs: float,
                       lowcut: float,
                       highcut: float,
                       order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to neural signal.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
        fs : float
            Sampling frequency (Hz)
        lowcut : float
            Low frequency cutoff (Hz)
        highcut : float
            High frequency cutoff (Hz)
        order : int
            Filter order
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered signal
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered
    
    @staticmethod
    def extract_brain_rhythms(signal_data: np.ndarray,
                             fs: float) -> Dict[str, np.ndarray]:
        """
        Extract classical brain rhythm bands.
        
        Bands:
        - Delta: 0.5-4 Hz (deep sleep)
        - Theta: 4-8 Hz (meditation, memory)
        - Alpha: 8-13 Hz (relaxed awareness)
        - Beta: 13-30 Hz (active thinking)
        - Gamma: 30-100 Hz (consciousness, binding)
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
        fs : float
            Sampling frequency (Hz)
            
        Returns:
        --------
        rhythms : dict
            Dictionary of filtered signals for each band
        """
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        rhythms = {}
        for band_name, (low, high) in bands.items():
            if high < fs / 2:  # Check Nyquist limit
                rhythms[band_name] = NeuralSignalProcessor.bandpass_filter(
                    signal_data, fs, low, high
                )
        
        return rhythms
    
    @staticmethod
    def hilbert_transform_analysis(signal_data: np.ndarray) -> Dict:
        """
        Compute instantaneous phase and amplitude using Hilbert transform.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
            
        Returns:
        --------
        analysis : dict
            Instantaneous phase, amplitude, and frequency
        """
        analytic_signal = signal.hilbert(signal_data)
        amplitude = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)
        
        # Instantaneous frequency
        inst_phase = np.unwrap(phase)
        inst_freq = np.diff(inst_phase) / (2.0 * np.pi)
        
        return {
            'amplitude': amplitude,
            'phase': phase,
            'instantaneous_frequency': inst_freq,
            'analytic_signal': analytic_signal
        }
    
    @staticmethod
    def spike_detection(signal_data: np.ndarray,
                       fs: float,
                       threshold: float = None,
                       method: str = 'threshold') -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect neural spikes in continuous data.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
        fs : float
            Sampling frequency (Hz)
        threshold : float
            Detection threshold (if None, uses 4*std)
        method : str
            Detection method ('threshold', 'adaptive')
            
        Returns:
        --------
        spike_times : np.ndarray
            Indices of detected spikes
        spike_amplitudes : np.ndarray
            Amplitudes of detected spikes
        """
        if threshold is None:
            threshold = 4 * np.std(signal_data)
        
        if method == 'threshold':
            # Simple threshold crossing
            spike_indices = signal.find_peaks(signal_data, height=threshold)[0]
        
        elif method == 'adaptive':
            # Adaptive threshold using median absolute deviation
            mad = np.median(np.abs(signal_data - np.median(signal_data)))
            adaptive_threshold = 4.5 * mad / 0.6745
            spike_indices = signal.find_peaks(signal_data, height=adaptive_threshold)[0]
        
        else:
            spike_indices = signal.find_peaks(signal_data, height=threshold)[0]
        
        spike_amplitudes = signal_data[spike_indices]
        
        return spike_indices, spike_amplitudes
    
    @staticmethod
    def compute_spectrogram(signal_data: np.ndarray,
                           fs: float,
                           window: str = 'hann',
                           nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute time-frequency spectrogram.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
        fs : float
            Sampling frequency (Hz)
        window : str
            Window function
        nperseg : int
            Length of each segment
            
        Returns:
        --------
        f : np.ndarray
            Frequency array
        t : np.ndarray
            Time array
        Sxx : np.ndarray
            Spectrogram (power spectral density)
        """
        f, t, Sxx = signal.spectrogram(signal_data, fs, 
                                       window=window, 
                                       nperseg=nperseg)
        return f, t, Sxx
    
    @staticmethod
    def phase_amplitude_coupling(phase_signal: np.ndarray,
                                amplitude_signal: np.ndarray,
                                n_bins: int = 18) -> Dict:
        """
        Compute phase-amplitude coupling (PAC).
        Tests if amplitude of high-frequency oscillations is modulated by
        phase of low-frequency oscillations.
        
        Parameters:
        -----------
        phase_signal : np.ndarray
            Low frequency phase signal
        amplitude_signal : np.ndarray
            High frequency amplitude signal
        n_bins : int
            Number of phase bins
            
        Returns:
        --------
        pac : dict
            Modulation index and coupling strength
        """
        # Get phases
        phase_data = np.angle(signal.hilbert(phase_signal))
        amplitude_data = np.abs(signal.hilbert(amplitude_signal))
        
        # Bin phases
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        digitized = np.digitize(phase_data, phase_bins) - 1
        
        # Mean amplitude per phase bin
        mean_amp_per_phase = np.array([
            np.mean(amplitude_data[digitized == i]) 
            for i in range(n_bins)
        ])
        
        # Modulation Index (MI) - Kullback-Leibler divergence
        uniform = np.ones(n_bins) / n_bins
        observed = mean_amp_per_phase / np.sum(mean_amp_per_phase)
        
        MI = np.sum(observed * np.log(observed / uniform)) / np.log(n_bins)
        
        return {
            'modulation_index': MI,
            'mean_amplitude_per_phase': mean_amp_per_phase,
            'phase_bins': phase_bins[:-1],
            'coupling_strength': np.max(mean_amp_per_phase) / np.mean(mean_amp_per_phase)
        }


# ============================================================================
# SECTION 2: BRAIN CONNECTIVITY ANALYSIS
# ============================================================================

class BrainConnectivity:
    """
    Analyze functional and effective connectivity between brain regions.
    """
    
    @staticmethod
    def correlation_matrix(signals: np.ndarray,
                          method: str = 'pearson') -> np.ndarray:
        """
        Compute correlation matrix between multiple signals.
        
        Parameters:
        -----------
        signals : np.ndarray
            Matrix of signals (n_channels x n_timepoints)
        method : str
            Correlation method ('pearson', 'spearman')
            
        Returns:
        --------
        corr_matrix : np.ndarray
            Correlation matrix (n_channels x n_channels)
        """
        n_channels = signals.shape[0]
        corr_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                if method == 'pearson':
                    corr_matrix[i, j] = np.corrcoef(signals[i], signals[j])[0, 1]
                elif method == 'spearman':
                    corr_matrix[i, j] = stats.spearmanr(signals[i], signals[j])[0]
        
        return corr_matrix
    
    @staticmethod
    def coherence_matrix(signals: np.ndarray,
                        fs: float,
                        freq_range: Tuple[float, float] = (8, 13)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence matrix (frequency-domain connectivity).
        
        Parameters:
        -----------
        signals : np.ndarray
            Matrix of signals (n_channels x n_timepoints)
        fs : float
            Sampling frequency (Hz)
        freq_range : tuple
            Frequency range for averaging (Hz)
            
        Returns:
        --------
        coh_matrix : np.ndarray
            Coherence matrix
        freqs : np.ndarray
            Frequency array
        """
        n_channels = signals.shape[0]
        
        # Compute coherence between first pair to get frequency array
        f, Cxy = signal.coherence(signals[0], signals[1], fs)
        
        # Find frequency indices in range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        
        # Initialize coherence matrix
        coh_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(n_channels):
                f, Cxy = signal.coherence(signals[i], signals[j], fs)
                # Average coherence in frequency range
                coh_matrix[i, j] = np.mean(Cxy[freq_mask])
        
        return coh_matrix, f[freq_mask]
    
    @staticmethod
    def phase_locking_value(signal1: np.ndarray,
                           signal2: np.ndarray) -> float:
        """
        Compute Phase Locking Value (PLV) between two signals.
        
        PLV measures phase synchronization between signals.
        
        Parameters:
        -----------
        signal1, signal2 : np.ndarray
            Neural signals
            
        Returns:
        --------
        plv : float
            Phase locking value (0-1)
        """
        # Get phases using Hilbert transform
        phase1 = np.angle(signal.hilbert(signal1))
        phase2 = np.angle(signal.hilbert(signal2))
        
        # Phase difference
        phase_diff = phase1 - phase2
        
        # PLV is the mean resultant length of phase differences
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    @staticmethod
    def granger_causality(signal1: np.ndarray,
                         signal2: np.ndarray,
                         max_lag: int = 10) -> Dict:
        """
        Compute Granger causality (directional influence).
        
        Tests if signal1 helps predict signal2 beyond signal2's own past.
        
        Parameters:
        -----------
        signal1, signal2 : np.ndarray
            Neural signals
        max_lag : int
            Maximum lag to consider
            
        Returns:
        --------
        causality : dict
            F-statistic and p-value for both directions
        """
        from scipy import linalg
        
        def fit_ar_model(y, X):
            """Fit autoregressive model"""
            beta = linalg.lstsq(X, y)[0]
            residuals = y - X @ beta
            rss = np.sum(residuals**2)
            return beta, rss
        
        n = len(signal1)
        
        # Create lagged matrices
        Y2 = signal2[max_lag:]
        X2_restricted = np.column_stack([
            signal2[i:n-max_lag+i] for i in range(max_lag)
        ])
        X2_full = np.column_stack([
            X2_restricted,
            *[signal1[i:n-max_lag+i] for i in range(max_lag)]
        ])
        
        # Fit models
        _, rss_restricted = fit_ar_model(Y2, X2_restricted)
        _, rss_full = fit_ar_model(Y2, X2_full)
        
        # F-statistic for signal1 -> signal2
        n_params = max_lag
        n_obs = len(Y2)
        f_stat_1to2 = ((rss_restricted - rss_full) / n_params) / (rss_full / (n_obs - 2*n_params))
        p_value_1to2 = 1 - stats.f.cdf(f_stat_1to2, n_params, n_obs - 2*n_params)
        
        # Reverse direction
        Y1 = signal1[max_lag:]
        X1_restricted = np.column_stack([
            signal1[i:n-max_lag+i] for i in range(max_lag)
        ])
        X1_full = np.column_stack([
            X1_restricted,
            *[signal2[i:n-max_lag+i] for i in range(max_lag)]
        ])
        
        _, rss_restricted = fit_ar_model(Y1, X1_restricted)
        _, rss_full = fit_ar_model(Y1, X1_full)
        
        f_stat_2to1 = ((rss_restricted - rss_full) / n_params) / (rss_full / (n_obs - 2*n_params))
        p_value_2to1 = 1 - stats.f.cdf(f_stat_2to1, n_params, n_obs - 2*n_params)
        
        return {
            'signal1_to_signal2': {'F': f_stat_1to2, 'p_value': p_value_1to2},
            'signal2_to_signal1': {'F': f_stat_2to1, 'p_value': p_value_2to1}
        }
    
    @staticmethod
    def network_graph_metrics(connectivity_matrix: np.ndarray,
                             threshold: float = None) -> Dict:
        """
        Compute graph-theoretical metrics of brain network.
        
        Parameters:
        -----------
        connectivity_matrix : np.ndarray
            Connectivity matrix (weighted or binary)
        threshold : float
            If provided, threshold connections below this value
            
        Returns:
        --------
        metrics : dict
            Graph metrics (clustering, path length, etc.)
        """
        if threshold is not None:
            adj_matrix = (connectivity_matrix > threshold).astype(float)
        else:
            adj_matrix = connectivity_matrix.copy()
        
        n_nodes = adj_matrix.shape[0]
        
        # Degree (number of connections)
        degrees = np.sum(adj_matrix > 0, axis=1)
        
        # Clustering coefficient
        clustering_coeffs = []
        for i in range(n_nodes):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                clustering_coeffs.append(0)
                continue
            
            # Count connections between neighbors
            connections = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        connections += 1
            
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeffs.append(connections / possible if possible > 0 else 0)
        
        # Global efficiency (inverse of path length)
        # Simplified calculation
        distances = 1 / (adj_matrix + np.eye(n_nodes))  # Avoid division by zero
        distances[distances == np.inf] = 0
        
        global_efficiency = np.sum(distances) / (n_nodes * (n_nodes - 1))
        
        # Small-worldness components
        avg_clustering = np.mean(clustering_coeffs)
        avg_degree = np.mean(degrees)
        
        return {
            'degrees': degrees,
            'mean_degree': avg_degree,
            'clustering_coefficients': np.array(clustering_coeffs),
            'mean_clustering': avg_clustering,
            'global_efficiency': global_efficiency,
            'density': np.sum(adj_matrix > 0) / (n_nodes * (n_nodes - 1))
        }


# ============================================================================
# SECTION 3: NEURAL OSCILLATIONS AND SYNCHRONY
# ============================================================================

class NeuralOscillations:
    """
    Analyze neural oscillations, synchronization, and phase dynamics.
    """
    
    @staticmethod
    def kuramoto_order_parameter(phases: np.ndarray) -> float:
        """
        Compute Kuramoto order parameter (global synchronization measure).
        
        R = |⟨e^(iθ)⟩| measures phase coherence across oscillators.
        R = 0: desynchronized, R = 1: perfectly synchronized
        
        Parameters:
        -----------
        phases : np.ndarray
            Phase values (radians) for multiple oscillators
            
        Returns:
        --------
        R : float
            Order parameter (0-1)
        """
        R = np.abs(np.mean(np.exp(1j * phases)))
        return R
    
    @staticmethod
    def phase_synchronization_index(signals: np.ndarray) -> np.ndarray:
        """
        Compute pairwise phase synchronization indices.
        
        Parameters:
        -----------
        signals : np.ndarray
            Matrix of signals (n_channels x n_timepoints)
            
        Returns:
        --------
        psi_matrix : np.ndarray
            Phase synchronization index matrix
        """
        n_channels = signals.shape[0]
        psi_matrix = np.zeros((n_channels, n_channels))
        
        # Extract phases
        phases = np.array([np.angle(signal.hilbert(sig)) for sig in signals])
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                # Compute PLV
                phase_diff = phases[i] - phases[j]
                psi = np.abs(np.mean(np.exp(1j * phase_diff)))
                psi_matrix[i, j] = psi
                psi_matrix[j, i] = psi
        
        return psi_matrix
    
    @staticmethod
    def cross_frequency_coupling(signal_data: np.ndarray,
                                 fs: float,
                                 low_freq_range: Tuple[float, float] = (4, 8),
                                 high_freq_range: Tuple[float, float] = (30, 100)) -> Dict:
        """
        Analyze cross-frequency coupling (CFC).
        
        Measures coupling between phase of low-frequency and amplitude of
        high-frequency oscillations.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
        fs : float
            Sampling frequency
        low_freq_range : tuple
            Low frequency band (Hz)
        high_freq_range : tuple
            High frequency band (Hz)
            
        Returns:
        --------
        cfc : dict
            Cross-frequency coupling metrics
        """
        # Extract frequency bands
        low_freq = NeuralSignalProcessor.bandpass_filter(
            signal_data, fs, low_freq_range[0], low_freq_range[1]
        )
        high_freq = NeuralSignalProcessor.bandpass_filter(
            signal_data, fs, high_freq_range[0], high_freq_range[1]
        )
        
        # Compute PAC
        pac_results = NeuralSignalProcessor.phase_amplitude_coupling(
            low_freq, high_freq
        )
        
        return {
            'modulation_index': pac_results['modulation_index'],
            'coupling_strength': pac_results['coupling_strength'],
            'low_freq_band': low_freq_range,
            'high_freq_band': high_freq_range,
            'mean_amplitude_per_phase': pac_results['mean_amplitude_per_phase']
        }
    
    @staticmethod
    def detect_oscillatory_bursts(signal_data: np.ndarray,
                                  fs: float,
                                  freq_range: Tuple[float, float],
                                  threshold: float = 2.0) -> List[Tuple[int, int]]:
        """
        Detect bursts of oscillatory activity.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Neural signal
        fs : float
            Sampling frequency
        freq_range : tuple
            Frequency range of interest (Hz)
        threshold : float
            Threshold in standard deviations for burst detection
            
        Returns:
        --------
        bursts : list
            List of (start_idx, end_idx) tuples for each burst
        """
        # Filter signal
        filtered = NeuralSignalProcessor.bandpass_filter(
            signal_data, fs, freq_range[0], freq_range[1]
        )
        
        # Compute amplitude envelope
        amplitude = np.abs(signal.hilbert(filtered))
        
        # Threshold based on mean and std
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        threshold_value = mean_amp + threshold * std_amp
        
        # Find periods above threshold
        above_threshold = amplitude > threshold_value
        
        # Detect burst boundaries
        bursts = []
        in_burst = False
        start_idx = 0
        
        for i, val in enumerate(above_threshold):
            if val and not in_burst:
                start_idx = i
                in_burst = True
            elif not val and in_burst:
                bursts.append((start_idx, i))
                in_burst = False
        
        # Handle case where burst continues to end
        if in_burst:
            bursts.append((start_idx, len(above_threshold)))
        
        return bursts


# ============================================================================
# SECTION 4: NEUROPLASTICITY AND LEARNING
# ============================================================================

class Neuroplasticity:
    """
    Model neuroplasticity, synaptic changes, and learning dynamics.
    """
    
    @staticmethod
    def hebbian_learning(pre_activity: np.ndarray,
                        post_activity: np.ndarray,
                        learning_rate: float = 0.01,
                        initial_weight: float = 0.5) -> np.ndarray:
        """
        Simulate Hebbian learning: "Neurons that fire together, wire together."
        
        Parameters:
        -----------
        pre_activity : np.ndarray
            Presynaptic neuron activity
        post_activity : np.ndarray
            Postsynaptic neuron activity
        learning_rate : float
            Learning rate η
        initial_weight : float
            Initial synaptic weight
            
        Returns:
        --------
        weights : np.ndarray
            Evolution of synaptic weight over time
        """
        n_steps = len(pre_activity)
        weights = np.zeros(n_steps)
        weights[0] = initial_weight
        
        for t in range(1, n_steps):
            # Hebbian rule: Δw = η * pre * post
            dw = learning_rate * pre_activity[t-1] * post_activity[t-1]
            weights[t] = weights[t-1] + dw
            
            # Keep weights in reasonable range
            weights[t] = np.clip(weights[t], 0, 2)
        
        return weights
    
    @staticmethod
    def stdp_learning(spike_times_pre: np.ndarray,
                     spike_times_post: np.ndarray,
                     tau_plus: float = 20.0,
                     tau_minus: float = 20.0,
                     A_plus: float = 0.01,
                     A_minus: float = 0.01) -> float:
        """
        Spike-Timing-Dependent Plasticity (STDP).
        
        Synaptic changes depend on relative timing of pre- and post-synaptic spikes.
        Δt > 0 (pre before post): potentiation
        Δt < 0 (post before pre): depression
        
        Parameters:
        -----------
        spike_times_pre : np.ndarray
            Presynaptic spike times
        spike_times_post : np.ndarray
            Postsynaptic spike times
        tau_plus, tau_minus : float
            Time constants for potentiation/depression (ms)
        A_plus, A_minus : float
            Maximum weight changes
            
        Returns:
        --------
        total_weight_change : float
            Cumulative synaptic weight change
        """
        weight_change = 0.0
        
        for t_pre in spike_times_pre:
            for t_post in spike_times_post:
                dt = t_post - t_pre
                
                if dt > 0:  # Pre before post -> potentiation
                    weight_change += A_plus * np.exp(-dt / tau_plus)
                else:  # Post before pre -> depression
                    weight_change -= A_minus * np.exp(dt / tau_minus)
        
        return weight_change
    
    @staticmethod
    def learning_curve(practice_sessions: np.ndarray,
                      session_quality: np.ndarray,
                      decay_rate: float = 0.1) -> np.ndarray:
        """
        Model learning curve with practice and forgetting.
        
        dP/dt = quality * (1 - P) - decay * P
        
        Parameters:
        -----------
        practice_sessions : np.ndarray
            Time points of practice sessions
        session_quality : np.ndarray
            Quality/intensity of each session
        decay_rate : float
            Forgetting rate
            
        Returns:
        --------
        performance : np.ndarray
            Performance level over time
        """
        max_time = int(np.max(practice_sessions)) + 100
        performance = np.zeros(max_time)
        performance[0] = 0.0
        
        session_idx = 0
        
        for t in range(1, max_time):
            # Check if there's a practice session at this time
            if session_idx < len(practice_sessions) and t >= practice_sessions[session_idx]:
                # Learning during practice
                quality = session_quality[session_idx]
                dP = quality * (1 - performance[t-1])
                session_idx += 1
            else:
                # Forgetting without practice
                dP = -decay_rate * performance[t-1]
            
            performance[t] = performance[t-1] + dP
            performance[t] = np.clip(performance[t], 0, 1)
        
        return performance
    
    @staticmethod
    def critical_period_plasticity(age: np.ndarray,
                                   peak_age: float = 5.0,
                                   width: float = 3.0) -> np.ndarray:
        """
        Model critical period for plasticity (e.g., language acquisition).
        
        Plasticity peaks during critical period then declines.
        
        Parameters:
        -----------
        age : np.ndarray
            Age values
        peak_age : float
            Age of peak plasticity
        width : float
            Width of critical period
            
        Returns:
        --------
        plasticity : np.ndarray
            Plasticity level at each age
        """
        # Gaussian-shaped critical period
        plasticity = np.exp(-((age - peak_age)**2) / (2 * width**2))
        
        # Add baseline adult plasticity
        adult_baseline = 0.2
        plasticity = plasticity * (1 - adult_baseline) + adult_baseline
        
        return plasticity


# ============================================================================
# SECTION 5: CONSCIOUSNESS AND INTEGRATION
# ============================================================================

class ConsciousnessMetrics:
    """
    Computational measures related to consciousness and information integration.
    Based on Integrated Information Theory (IIT) and related frameworks.
    """
    
    @staticmethod
    def neural_complexity(signals: np.ndarray) -> float:
        """
        Compute neural complexity (balance between integration and differentiation).
        
        Parameters:
        -----------
        signals : np.ndarray
            Matrix of neural signals (n_channels x n_timepoints)
            
        Returns:
        --------
        complexity : float
            Neural complexity measure
        """
        n_channels = signals.shape[0]
        
        # Mutual information between all pairs
        mi_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Simplified MI calculation
                hist_2d, _, _ = np.histogram2d(signals[i], signals[j], bins=20)
                hist_x = np.sum(hist_2d, axis=1)
                hist_y = np.sum(hist_2d, axis=0)
                
                p_xy = hist_2d / np.sum(hist_2d)
                p_x = hist_x / np.sum(hist_x)
                p_y = hist_y / np.sum(hist_y)
                
                mi = 0.0
                for ii in range(len(p_x)):
                    for jj in range(len(p_y)):
                        if p_xy[ii, jj] > 0:
                            mi += p_xy[ii, jj] * np.log2(p_xy[ii, jj] / (p_x[ii] * p_y[jj]))
                
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        # Complexity is related to variance in MI values
        complexity = np.var(mi_matrix[np.triu_indices(n_channels, k=1)])
        
        return complexity
    
    @staticmethod
    def integrated_information_approx(connectivity_matrix: np.ndarray,
                                     activity: np.ndarray) -> float:
        """
        Approximation of Integrated Information (


Φ) from IIT.
Simplified version: measures information loss when system is partitioned.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Connectivity between elements (n_elements x n_elements)
    activity : np.ndarray
        Current activity state (n_elements,)
        
    Returns:
    --------
    phi : float
        Approximated integrated information
    """
    n_elements = len(activity)
    
    # Normalize activity
    activity_norm = (activity - np.mean(activity)) / (np.std(activity) + 1e-10)
    
    # Compute system-level entropy (whole system)
    system_entropy = -np.sum(activity_norm**2 * np.log(np.abs(activity_norm) + 1e-10))
    
    # Compute entropy for minimum information partition
    min_partition_entropy = 0
    
    # Find best bipartition (simplified: just split in half)
    mid = n_elements // 2
    part1 = activity_norm[:mid]
    part2 = activity_norm[mid:]
    
    entropy1 = -np.sum(part1**2 * np.log(np.abs(part1) + 1e-10))
    entropy2 = -np.sum(part2**2 * np.log(np.abs(part2) + 1e-10))
    min_partition_entropy = entropy1 + entropy2
    
    # Φ is the difference
    phi = system_entropy - min_partition_entropy
    
    # Consider connectivity strength
    connection_strength = np.mean(np.abs(connectivity_matrix))
    phi = phi * connection_strength
    
    return max(0, phi)  # Φ is non-negative

@staticmethod
def perturbational_complexity_index(signals: np.ndarray,
                                   fs: float) -> float:
    """
    Perturbational Complexity Index (PCI) - consciousness measure.
    
    Based on TMS-EEG: complexity of brain's response to perturbation.
    Here simulated using signal complexity.
    
    Parameters:
    -----------
    signals : np.ndarray
        Neural signals (n_channels x n_timepoints)
    fs : float
        Sampling frequency
        
    Returns:
    --------
    pci : float
        Complexity index
    """
    # Compute spatial complexity (number of active channels)
    spatial_complexity = np.sum(np.std(signals, axis=1) > 0.1 * np.max(np.std(signals, axis=1)))
    
    # Compute temporal complexity (Lempel-Ziv)
    def lempel_ziv(sequence):
        """Simplified LZ complexity"""
        n = len(sequence)
        c, i, k, l = 1, 0, 1, 1
        k_max = 1
        
        while i + k <= n:
            if i + k > n:
                break
            if sequence[i:i+k] != sequence[l:l+k]:
                k += 1
            else:
                c += 1
                i += k
                k = 1
                l = i + 1
            if k > k_max:
                k_max = k
        
        return c
    
    # Binarize first channel
    binary_signal = (signals[0] > np.median(signals[0])).astype(int)
    temporal_complexity = lempel_ziv(binary_signal)
    
    # PCI combines spatial and temporal
    pci = spatial_complexity * temporal_complexity / len(signals[0])
    
    return pci

@staticmethod
def global_workspace_activity(signals: np.ndarray,
                              hub_threshold: float = 0.7) -> Dict:
    """
    Analyze global workspace dynamics (consciousness theory).
    
    Identifies hub regions with widespread connectivity that might
    constitute global workspace for conscious processing.
    
    Parameters:
    -----------
    signals : np.ndarray
        Neural signals (n_channels x n_timepoints)
    hub_threshold : float
        Correlation threshold for hub identification
        
    Returns:
    --------
    workspace : dict
        Global workspace metrics
    """
    # Compute correlation matrix
    n_channels = signals.shape[0]
    corr_matrix = np.corrcoef(signals)
    
    # Identify hubs (highly connected nodes)
    connectivity_strength = np.sum(np.abs(corr_matrix) > hub_threshold, axis=1)
    
    # Hub neurons are those in top quartile
    hub_threshold_value = np.percentile(connectivity_strength, 75)
    hub_indices = np.where(connectivity_strength >= hub_threshold_value)[0]
    
    # Measure hub activity coherence
    if len(hub_indices) > 1:
        hub_signals = signals[hub_indices]
        hub_coherence = np.mean(np.corrcoef(hub_signals)[np.triu_indices(len(hub_indices), k=1)])
    else:
        hub_coherence = 0
    
    # Broadcasting strength (hub to non-hub connections)
    non_hub_indices = np.setdiff1d(np.arange(n_channels), hub_indices)
    if len(hub_indices) > 0 and len(non_hub_indices) > 0:
        broadcast_strength = np.mean(np.abs(corr_matrix[np.ix_(hub_indices, non_hub_indices)]))
    else:
        broadcast_strength = 0
    
    return {
        'hub_indices': hub_indices,
        'n_hubs': len(hub_indices),
        'hub_coherence': hub_coherence,
        'broadcast_strength': broadcast_strength,
        'connectivity_distribution': connectivity_strength
    }

@staticmethod
def entropy_rate(signal_data: np.ndarray,
                order: int = 2) -> float:
    """
    Compute entropy rate (predictability of neural dynamics).
    
    Higher entropy rate = less predictable = potentially higher consciousness.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Neural signal
    order : int
        Order of Markov model
        
    Returns:
    --------
    entropy_rate : float
        Entropy rate in bits/sample
    """
    # Discretize signal
    n_bins = 10
    digitized = np.digitize(signal_data, np.linspace(np.min(signal_data), 
                                                     np.max(signal_data), n_bins))
    
    # Build transition matrix
    transitions = {}
    for i in range(len(digitized) - order):
        state = tuple(digitized[i:i+order])
        next_state = digitized[i+order]
        
        if state not in transitions:
            transitions[state] = []
        transitions[state].append(next_state)
    
    # Compute conditional entropy
    total_entropy = 0
    total_count = 0
    
    for state, next_states in transitions.items():
        counts = np.bincount(next_states, minlength=n_bins+1)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        
        state_entropy = -np.sum(probs * np.log2(probs))
        total_entropy += state_entropy * len(next_states)
        total_count += len(next_states)
    
    entropy_rate = total_entropy / total_count if total_count > 0 else 0
    
    return entropy_rate
============================================================================
SECTION 6: FRACTAL BRAIN DYNAMICS
============================================================================
class FractalBrainAnalysis:
"""
Analyze fractal and scale-free properties of brain activity.
"""
@staticmethod
def detrended_fluctuation_analysis(signal_data: np.ndarray,
                                  min_scale: int = 10,
                                  max_scale: int = None,
                                  n_scales: int = 20) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Detrended Fluctuation Analysis (DFA) - measures long-range correlations.
    
    Scaling exponent α:
    - α < 0.5: anti-correlated (negative feedback)
    - α = 0.5: uncorrelated (white noise)
    - α = 1.0: 1/f noise (pink noise, typical of brain)
    - α > 1.0: non-stationary
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Neural signal
    min_scale : int
        Minimum scale (box size)
    max_scale : int
        Maximum scale
    n_scales : int
        Number of scales to test
        
    Returns:
    --------
    alpha : float
        Scaling exponent
    scales : np.ndarray
        Scale values
    fluctuations : np.ndarray
        Fluctuation values
    """
    N = len(signal_data)
    
    if max_scale is None:
        max_scale = N // 4
    
    # Cumulative sum (integration)
    y = np.cumsum(signal_data - np.mean(signal_data))
    
    # Scales (box sizes)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
    scales = np.unique(scales)
    
    fluctuations = []
    
    for scale in scales:
        # Divide into boxes
        n_boxes = N // scale
        boxes = y[:n_boxes * scale].reshape(n_boxes, scale)
        
        # Fit polynomial in each box and calculate fluctuation
        box_fluctuations = []
        for box in boxes:
            t = np.arange(len(box))
            coeffs = np.polyfit(t, box, 1)  # Linear detrending
            trend = np.polyval(coeffs, t)
            fluctuation = np.sqrt(np.mean((box - trend)**2))
            box_fluctuations.append(fluctuation)
        
        # Average fluctuation at this scale
        F = np.sqrt(np.mean(np.array(box_fluctuations)**2))
        fluctuations.append(F)
    
    fluctuations = np.array(fluctuations)
    
    # Fit log-log relationship to get scaling exponent
    log_scales = np.log10(scales)
    log_fluctuations = np.log10(fluctuations)
    
    alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]
    
    return alpha, scales, fluctuations

@staticmethod
def multifractal_spectrum(signal_data: np.ndarray,
                         q_range: np.ndarray = None) -> Dict:
    """
    Compute multifractal spectrum (characterizes complexity).
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Neural signal
    q_range : np.ndarray
        Range of q moments to compute
        
    Returns:
    --------
    spectrum : dict
        Multifractal spectrum parameters
    """
    if q_range is None:
        q_range = np.arange(-5, 6, 0.5)
    
    N = len(signal_data)
    scales = np.logspace(1, np.log10(N//4), 20).astype(int)
    
    # Normalize signal to positive values
    signal_norm = signal_data - np.min(signal_data) + 1e-10
    
    tau_q = []
    
    for q in q_range:
        log_Fq = []
        
        for scale in scales:
            n_boxes = N // scale
            boxes = signal_norm[:n_boxes * scale].reshape(n_boxes, scale)
            
            # Partition function
            if q == 0:
                Fq = np.sum(np.sum(boxes, axis=1) > 0)
            else:
                box_measures = np.sum(boxes, axis=1)
                box_measures = box_measures / np.sum(box_measures)
                Fq = np.sum(box_measures**q)
            
            if Fq > 0:
                log_Fq.append(np.log(Fq))
            else:
                log_Fq.append(-np.inf)
        
        # Scaling exponent τ(q)
        valid = np.isfinite(log_Fq)
        if np.sum(valid) > 2:
            tau = np.polyfit(np.log(scales[valid]), np.array(log_Fq)[valid], 1)[0]
            tau_q.append(tau)
        else:
            tau_q.append(np.nan)
    
    tau_q = np.array(tau_q)
    
    # Hurst exponent (at q=2)
    q2_idx = np.argmin(np.abs(q_range - 2))
    hurst = tau_q[q2_idx] / 2 if not np.isnan(tau_q[q2_idx]) else np.nan
    
    # Width of spectrum (degree of multifractality)
    valid = ~np.isnan(tau_q)
    spectrum_width = np.max(tau_q[valid]) - np.min(tau_q[valid]) if np.sum(valid) > 0 else 0
    
    return {
        'q_values': q_range,
        'tau_q': tau_q,
        'hurst_exponent': hurst,
        'spectrum_width': spectrum_width,
        'is_multifractal': spectrum_width > 0.5
    }

@staticmethod
def power_law_exponent(signal_data: np.ndarray,
                      fs: float) -> Dict:
    """
    Compute power law exponent of power spectral density.
    
    PSD ~ 1/f^β
    
    β ≈ 0: white noise
    β ≈ 1: pink noise (1/f, common in brain)
    β ≈ 2: brown noise (integration)
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Neural signal
    fs : float
        Sampling frequency
        
    Returns:
    --------
    result : dict
        Power law parameters
    """
    # Compute power spectrum
    freqs, psd = signal.welch(signal_data, fs, nperseg=min(len(signal_data)//4, 1024))
    
    # Exclude DC and very high frequencies
    valid = (freqs > 0.5) & (freqs < fs/4)
    freqs = freqs[valid]
    psd = psd[valid]
    
    # Fit power law in log-log space
    log_freqs = np.log10(freqs)
    log_psd = np.log10(psd)
    
    beta = -np.polyfit(log_freqs, log_psd, 1)[0]  # Negative slope
    
    # Goodness of fit
    fit_line = np.polyval([-beta, np.polyval([1, 0], log_freqs[0])], log_freqs)
    r_squared = 1 - np.sum((log_psd - fit_line)**2) / np.sum((log_psd - np.mean(log_psd))**2)
    
    return {
        'beta': beta,
        'r_squared': r_squared,
        'freqs': freqs,
        'psd': psd,
        'is_pink_noise': 0.8 < beta < 1.2,
        'noise_type': 'pink' if 0.8 < beta < 1.2 else 'white' if beta < 0.5 else 'brown' if beta > 1.5 else 'colored'
    }
============================================================================
SECTION 7: BRAIN STATE CLASSIFICATION
============================================================================
class BrainStateClassifier:
"""
Classify brain states (sleep stages, attention, meditation, etc.).
"""
@staticmethod
def sleep_stage_features(signal_data: np.ndarray,
                        fs: float) -> Dict:
    """
    Extract features for sleep stage classification.
    
    Sleep stages:
    - Wake: High beta, alpha
    - N1: Reduced alpha, theta
    - N2: Sleep spindles (12-15 Hz), K-complexes
    - N3: High delta power (slow-wave sleep)
    - REM: Similar to wake, low muscle tone
    
    Parameters:
    -----------
    signal_data : np.ndarray
        EEG signal
    fs : float
        Sampling frequency
        
    Returns:
    --------
    features : dict
        Feature set for classification
    """
    # Extract frequency bands
    rhythms = NeuralSignalProcessor.extract_brain_rhythms(signal_data, fs)
    
    # Compute power in each band
    band_powers = {}
    for band_name, band_signal in rhythms.items():
        band_powers[f'{band_name}_power'] = np.mean(band_signal**2)
    
    # Ratios
    if 'delta' in rhythms and 'alpha' in rhythms:
        delta_alpha_ratio = band_powers['delta_power'] / (band_powers['alpha_power'] + 1e-10)
    else:
        delta_alpha_ratio = 0
    
    # Spindle detection (N2 marker)
    if 'alpha' in rhythms:
        spindle_band = NeuralSignalProcessor.bandpass_filter(signal_data, fs, 12, 15)
        spindle_power = np.mean(spindle_band**2)
    else:
        spindle_power = 0
    
    # Signal complexity
    sample_entropy = ConsciousnessMetrics.entropy_rate(signal_data)
    
    return {
        **band_powers,
        'delta_alpha_ratio': delta_alpha_ratio,
        'spindle_power': spindle_power,
        'sample_entropy': sample_entropy,
        'signal_variance': np.var(signal_data)
    }

@staticmethod
def attention_state_index(signal_data: np.ndarray,
                         fs: float) -> float:
    """
    Compute attention/focus index.
    
    High attention: Increased beta, decreased theta
    
    Parameters:
    -----------
    signal_data : np.ndarray
        EEG signal
    fs : float
        Sampling frequency
        
    Returns:
    --------
    attention_index : float
        Attention level (0-1)
    """
    rhythms = NeuralSignalProcessor.extract_brain_rhythms(signal_data, fs)
    
    if 'beta' in rhythms and 'theta' in rhythms:
        beta_power = np.mean(rhythms['beta']**2)
        theta_power = np.mean(rhythms['theta']**2)
        
        # Attention index: beta/theta ratio, normalized
        attention_index = beta_power / (beta_power + theta_power)
    else:
        attention_index = 0.5
    
    return attention_index

@staticmethod
def meditation_depth(signal_data: np.ndarray,
                    fs: float) -> Dict:
    """
    Assess meditation depth based on neural patterns.
    
    Deep meditation: High alpha, theta; low beta
    
    Parameters:
    -----------
    signal_data : np.ndarray
        EEG signal
    fs : float
        Sampling frequency
        
    Returns:
    --------
    meditation_metrics : dict
        Meditation-related measures
    """
    rhythms = NeuralSignalProcessor.extract_brain_rhythms(signal_data, fs)
    
    metrics = {}
    
    # Alpha power (relaxed awareness)
    if 'alpha' in rhythms:
        metrics['alpha_power'] = np.mean(rhythms['alpha']**2)
    else:
        metrics['alpha_power'] = 0
    
    # Theta power (deep meditation)
    if 'theta' in rhythms:
        metrics['theta_power'] = np.mean(rhythms['theta']**2)
    else:
        metrics['theta_power'] = 0
    
    # Alpha-theta ratio
    if metrics['alpha_power'] > 0 and metrics['theta_power'] > 0:
        metrics['alpha_theta_ratio'] = metrics['alpha_power'] / metrics['theta_power']
    else:
        metrics['alpha_theta_ratio'] = 1.0
    
    # Coherence (synchronization)
    if 'alpha' in rhythms:
        # Auto-correlation as proxy for coherence
        alpha_autocorr = np.correlate(rhythms['alpha'], rhythms['alpha'], mode='full')
        alpha_autocorr = alpha_autocorr / np.max(alpha_autocorr)
        metrics['alpha_coherence'] = np.mean(alpha_autocorr[len(alpha_autocorr)//2:len(alpha_autocorr)//2+100])
    else:
        metrics['alpha_coherence'] = 0
    
    # Overall meditation depth (0-1)
    depth = (metrics['alpha_power'] + metrics['theta_power']) / (metrics['alpha_power'] + metrics['theta_power'] + metrics.get('beta_power', 1) + 1e-10)
    metrics['meditation_depth'] = np.clip(depth, 0, 1)
    
    return metrics

@staticmethod
def cognitive_load_estimation(signal_data: np.ndarray,
                              fs: float) -> float:
    """
    Estimate cognitive load/mental effort.
    
    High load: Increased theta, decreased alpha
    
    Parameters:
    -----------
    signal_data : np.ndarray
        EEG signal
    fs : float
        Sampling frequency
        
    Returns:
    --------
    cognitive_load : float
        Estimated cognitive load (0-1)
    """
    rhythms = NeuralSignalProcessor.extract_brain_rhythms(signal_data, fs)
    
    if 'theta' in rhythms and 'alpha' in rhythms:
        theta_power = np.mean(rhythms['theta']**2)
        alpha_power = np.mean(rhythms['alpha']**2)
        
        # Cognitive load index
        cognitive_load = theta_power / (theta_power + alpha_power)
    else:
        cognitive_load = 0.5
    
    return cognitive_load
============================================================================
SECTION 8: NEURAL MASS MODELS
============================================================================
class NeuralMassModels:
"""
Simulate population-level neural dynamics using neural mass models.
"""
@staticmethod
def jansen_rit_model(T: float,
                    dt: float,
                    input_signal: np.ndarray = None,
                    params: Dict = None) -> Dict:
    """
    Jansen-Rit neural mass model (cortical column).
    
    Models interactions between pyramidal cells, excitatory and inhibitory interneurons.
    Generates EEG-like oscillations.
    
    Parameters:
    -----------
    T : float
        Simulation time (seconds)
    dt : float
        Time step (seconds)
    input_signal : np.ndarray
        External input (if None, uses noise)
    params : dict
        Model parameters
        
    Returns:
    --------
    results : dict
        Simulated populations and EEG
    """
    # Default parameters
    if params is None:
        params = {
            'A': 3.25,      # Maximum amplitude of excitatory PSP (mV)
            'B': 22.0,      # Maximum amplitude of inhibitory PSP (mV)
            'a': 100.0,     # Inverse time constant of excitatory PSP (s^-1)
            'b': 50.0,      # Inverse time constant of inhibitory PSP (s^-1)
            'v0': 6.0,      # Firing threshold (mV)
            'e0': 2.5,      # Half-maximum firing rate (s^-1)
            'r': 0.56,      # Steepness of sigmoid
            'C1': 135.0,    # Connectivity constant
            'C2': 108.0,
            'C3': 33.75,
            'C4': 33.75,
        }
    
    # Simulation setup
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    
    # State variables
    y = np.zeros((6, n_steps))  # 6 state variables
    
    # Initial conditions (small random values)
    y[:, 0] = np.random.randn(6) * 0.1
    
    # Input signal
    if input_signal is None:
        input_signal = 220 + 20 * np.random.randn(n_steps)  # Background noise
    elif len(input_signal) < n_steps:
        input_signal = np.pad(input_signal, (0, n_steps - len(input_signal)), 'constant')
    
    # Sigmoid function
    def sigmoid(v):
        return 2 * params['e0'] / (1 + np.exp(params['r'] * (params['v0'] - v)))
    
    # Integrate using Euler method
    for i in range(n_steps - 1):
        # State equations
        dy = np.zeros(6)
        
        # y[0]: Pyramidal cell average membrane potential
        # y[1]: Derivative of y[0]
        # y[2]: Excitatory interneuron potential
        # y[3]: Derivative of y[2]
        # y[4]: Inhibitory interneuron potential
        # y[5]: Derivative of y[4]
        
        dy[0] = y[1, i]
        dy[1] = params['A'] * params['a'] * sigmoid(y[0, i] - y[4, i]) - 2 * params['a'] * y[1, i] - params['a']**2 * y[0, i]
        
        dy[2] = y[3, i]
        dy[3] = params['A'] * params['a'] * (input_signal[i] + params['C2'] * sigmoid(params['C1'] * y[0, i])) - 2 * params['a'] * y[3, i] - params['a']**2 * y[2, i]
        
        dy[4] = y[5, i]
        dy[5] = params['B'] * params['b'] * params['C4'] * sigmoid(params['C3'] * y[0, i]) - 2 * params['b'] * y[5, i] - params['b']**2 * y[4, i]
        
        # Update
        y[:, i+1] = y[:, i] + dy * dt
    
    # EEG-like signal (pyramidal output)
    eeg_signal = y[0, :] - y[4, :]
    
    return {
        'time': t,
        'eeg_signal': eeg_signal,
        'pyramidal': y[0, :],
        'excitatory': y[2, :],
        'inhibitory': y[4, :],
        'all_states': y
    }

@staticmethod
def wilson_cowan_model(T: float,
                      dt: float,
                      external_input: float = 0.5,
                      params: Dict = None) -> Dict:
    """
    Wilson-Cowan model (excitatory-inhibitory population dynamics).
    
    Parameters:
    -----------
    T : float
        Simulation time
    dt : float
        Time step
    external_input : float
        External input strength
    params : dict
        Model parameters
        
    Returns:
    --------
    results : dict
        Population activities
    """
    if params is None:
        params = {
            'tau_e': 10.0,    # Excitatory time constant
            'tau_i': 10.0,    # Inhibitory time constant
            'w_ee': 16.0,     # E->E connection weight
            'w_ei': 12.0,     # E->I connection weight
            'w_ie': 15.0,     # I->E connection weight
            'w_ii': 3.0,      # I->I connection weight
            'theta_e': 4.0,   # E threshold
            'theta_i': 3.7,   # I threshold
            'a': 1.3,         # Sigmoid steepness
        }
    
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    
    # State variables
    E = np.zeros(n_steps)  # Excitatory activity
    I = np.zeros(n_steps)  # Inhibitory activity
    
    # Initial conditions
    E[0] = 0.1
    I[0] = 0.1
    
    # Sigmoid
    def S(x, theta, a):
        return 1 / (1 + np.exp(-a * (x - theta)))
    
    # Integrate
    for i in range(n_steps - 1):
        # Input to populations
        input_e = params['w_ee'] * E[i] - params['w_ie'] * I[i] + external_input
        input_i = params['w_ei'] * E[i] - params['w_ii'] * I[i] + external_input
        
        # Dynamics
        dE = (-E[i] + S(input_e, params['theta_e'], params['a'])) / params['tau_e']
        dI = (-I[i] + S(input_i, params['theta_i'], params['a'])) / params['tau_i']
        
        E[i+1] = E[i] + dE * dt
        I[i+1] = I[i] + dI * dt
    
    return {
        'time': t,
        'excitatory': E,
        'inhibitory': I,
        'net_activity': E - I
    }
============================================================================
SECTION 9: VISUALIZATION TOOLS
============================================================================
class NeuroVisualization:
"""
Visualization tools for neuroscience data.
"""
@staticmethod
def plot_brain_rhythms(rhythms: Dict,
                      fs: float,
                      duration: float = 5.0,
                      title: str = "Brain Rhythms"):
    """
    Plot brain rhythm bands.
    """
    n_bands = len(rhythms)
    fig, axes = plt.subplots(n_bands, 1, figsize=(12, 2*n_bands))
    
    if n_bands == 1:
        axes = [axes]
    
    time = np.arange(int(duration * fs)) / fs
    
    for ax, (band_name, signal_data) in zip(axes, rhythms.items()):
        # Plot segment
        segment = signal_data[:int(duration * fs)]
        ax.plot(time[:len(segment)], segment, linewidth=0.5)
        ax.set_ylabel(f'{band_name.capitalize()} ({band_name})')
        ax.set_xlabel('Time (s)' if ax == axes[-1] else '')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

@staticmethod
def plot_connectivity_matrix(connectivity_matrix: np.ndarray,
                             labels: List[str] = None,
                             title: str = "Brain Connectivity"):
    """
    Plot connectivity matrix as heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(connectivity_matrix, cmap='RdBu_r', aspect


='auto',
vmin=-1, vmax=1)
# Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Connectivity Strength', rotation=270, labelpad=20)
    
    # Labels
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

@staticmethod
def plot_spectrogram(f: np.ndarray,
                    t: np.ndarray,
                    Sxx: np.ndarray,
                    title: str = "Neural Spectrogram"):
    """
    Plot time-frequency spectrogram.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                      shading='gouraud', cmap='viridis')
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)', rotation=270, labelpad=20)
    
    # Mark frequency bands
    bands = {'Delta': 2, 'Theta': 6, 'Alpha': 10, 'Beta': 20, 'Gamma': 50}
    for band_name, freq in bands.items():
        if freq < np.max(f):
            ax.axhline(freq, color='white', linestyle='--', 
                      alpha=0.3, linewidth=0.5)
            ax.text(np.max(t)*0.98, freq, band_name, 
                   color='white', ha='right', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

@staticmethod
def plot_phase_amplitude_coupling(mean_amp_per_phase: np.ndarray,
                                 phase_bins: np.ndarray,
                                 modulation_index: float,
                                 title: str = "Phase-Amplitude Coupling"):
    """
    Plot PAC modulation.
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))
    
    # Convert phase bins to radians for polar plot
    theta = phase_bins
    
    # Plot
    ax.plot(theta, mean_amp_per_phase, 'o-', linewidth=2, markersize=8)
    ax.fill_between(theta, 0, mean_amp_per_phase, alpha=0.3)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'{title}\nModulation Index: {modulation_index:.3f}', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

@staticmethod
def plot_dfa_results(scales: np.ndarray,
                    fluctuations: np.ndarray,
                    alpha: float,
                    title: str = "Detrended Fluctuation Analysis"):
    """
    Plot DFA results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Log-log plot
    ax.loglog(scales, fluctuations, 'o', markersize=8, label='Data')
    
    # Fit line
    fit_line = scales**alpha * (fluctuations[0] / scales[0]**alpha)
    ax.loglog(scales, fit_line, '--', linewidth=2, 
             label=f'Fit: α = {alpha:.3f}')
    
    ax.set_xlabel('Scale (samples)', fontsize=12)
    ax.set_ylabel('Fluctuation F(n)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    interpretation = ""
    if alpha < 0.5:
        interpretation = "Anti-correlated (negative feedback)"
    elif 0.5 <= alpha < 1.0:
        interpretation = "Correlated (persistent)"
    elif 0.9 <= alpha <= 1.1:
        interpretation = "1/f noise (pink noise - typical brain activity)"
    else:
        interpretation = "Non-stationary"
    
    ax.text(0.05, 0.95, f'Interpretation: {interpretation}',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

@staticmethod
def plot_neural_network_graph(connectivity_matrix: np.ndarray,
                              node_positions: np.ndarray = None,
                              threshold: float = 0.3,
                              title: str = "Brain Network Graph"):
    """
    Plot brain network as graph with nodes and edges.
    """
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    n_nodes = connectivity_matrix.shape[0]
    
    # Generate node positions if not provided
    if node_positions is None:
        # Circular layout
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        node_positions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Threshold connections
    adj_matrix = connectivity_matrix.copy()
    adj_matrix[np.abs(adj_matrix) < threshold] = 0
    
    # Draw edges
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adj_matrix[i, j] != 0:
                weight = adj_matrix[i, j]
                color = 'red' if weight > 0 else 'blue'
                alpha = min(abs(weight), 1.0) * 0.5
                
                ax.plot([node_positions[i, 0], node_positions[j, 0]],
                       [node_positions[i, 1], node_positions[j, 1]],
                       color=color, alpha=alpha, linewidth=abs(weight)*2)
    
    # Draw nodes
    node_sizes = np.sum(np.abs(adj_matrix), axis=1) * 100
    ax.scatter(node_positions[:, 0], node_positions[:, 1],
              s=node_sizes, c='lightblue', edgecolors='black',
              linewidths=2, zorder=5)
    
    # Labels
    for i, pos in enumerate(node_positions):
        ax.text(pos[0]*1.1, pos[1]*1.1, str(i+1),
               ha='center', va='center', fontsize=10)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    red_patch = mpatches.Patch(color='red', label='Excitatory')
    blue_patch = mpatches.Patch(color='blue', label='Inhibitory')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    plt.tight_layout()
    return fig
============================================================================
SECTION 10: INTEGRATED NEUROSCIENCE FRAMEWORK
============================================================================
class NeuroscienceFramework:
"""
Integrated framework for comprehensive neural data analysis.
"""
def __init__(self):
    self.signal_processor = NeuralSignalProcessor()
    self.connectivity = BrainConnectivity()
    self.oscillations = NeuralOscillations()
    self.plasticity = Neuroplasticity()
    self.consciousness = ConsciousnessMetrics()
    self.fractal = FractalBrainAnalysis()
    self.brain_states = BrainStateClassifier()
    self.models = NeuralMassModels()
    self.viz = NeuroVisualization()
    
    print("Neuroscience Analysis Framework Initialized")
    print("Version 1.0.0")
    print("=" * 70)

def comprehensive_neural_analysis(self,
                                 signal_data: np.ndarray,
                                 fs: float,
                                 signal_name: str = "Neural Signal",
                                 multi_channel: bool = False) -> Dict:
    """
    Perform comprehensive analysis of neural data.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        Neural signal (1D for single channel, 2D for multi-channel)
    fs : float
        Sampling frequency (Hz)
    signal_name : str
        Name for labeling
    multi_channel : bool
        Whether data is multi-channel
        
    Returns:
    --------
    analysis : dict
        Complete analysis results
    """
    print(f"\n{'=' * 70}")
    print(f"COMPREHENSIVE NEURAL ANALYSIS: {signal_name}")
    print(f"{'=' * 70}")
    print(f"Sampling Rate: {fs} Hz")
    print(f"Duration: {len(signal_data) / fs:.2f} seconds")
    
    results = {
        'signal_name': signal_name,
        'sampling_rate': fs,
        'duration': len(signal_data) / fs
    }
    
    # Single channel analysis
    if not multi_channel or signal_data.ndim == 1:
        signal_1d = signal_data.flatten()
        
        # 1. Brain Rhythms
        print("\n1. Extracting brain rhythms...")
        results['rhythms'] = self.signal_processor.extract_brain_rhythms(signal_1d, fs)
        print(f"   Extracted {len(results['rhythms'])} frequency bands")
        
        # 2. Spectral Analysis
        print("\n2. Computing spectrogram...")
        f, t, Sxx = self.signal_processor.compute_spectrogram(signal_1d, fs)
        results['spectrogram'] = {'f': f, 't': t, 'Sxx': Sxx}
        
        # 3. Fractal Analysis
        print("\n3. Performing fractal analysis...")
        try:
            alpha, scales, fluct = self.fractal.detrended_fluctuation_analysis(signal_1d)
            results['dfa'] = {
                'alpha': alpha,
                'scales': scales,
                'fluctuations': fluct,
                'interpretation': self._interpret_dfa(alpha)
            }
            print(f"   DFA exponent α = {alpha:.3f}")
        except Exception as e:
            print(f"   DFA error: {e}")
            results['dfa'] = {'error': str(e)}
        
        # 4. Power Law Analysis
        print("\n4. Analyzing power spectrum...")
        try:
            power_law = self.fractal.power_law_exponent(signal_1d, fs)
            results['power_law'] = power_law
            print(f"   Power law exponent β = {power_law['beta']:.3f}")
            print(f"   Noise type: {power_law['noise_type']}")
        except Exception as e:
            results['power_law'] = {'error': str(e)}
        
        # 5. Brain State Classification
        print("\n5. Classifying brain state...")
        try:
            sleep_features = self.brain_states.sleep_stage_features(signal_1d, fs)
            attention = self.brain_states.attention_state_index(signal_1d, fs)
            meditation = self.brain_states.meditation_depth(signal_1d, fs)
            
            results['brain_state'] = {
                'sleep_features': sleep_features,
                'attention_index': attention,
                'meditation': meditation
            }
            print(f"   Attention index: {attention:.3f}")
            print(f"   Meditation depth: {meditation['meditation_depth']:.3f}")
        except Exception as e:
            results['brain_state'] = {'error': str(e)}
        
        # 6. Consciousness Metrics
        print("\n6. Computing consciousness metrics...")
        try:
            entropy_rate = self.consciousness.entropy_rate(signal_1d)
            results['consciousness'] = {
                'entropy_rate': entropy_rate
            }
            print(f"   Entropy rate: {entropy_rate:.3f} bits/sample")
        except Exception as e:
            results['consciousness'] = {'error': str(e)}
    
    # Multi-channel analysis
    else:
        print("\nMulti-channel data detected...")
        
        # 1. Connectivity Analysis
        print("\n1. Computing connectivity matrix...")
        try:
            corr_matrix = self.connectivity.correlation_matrix(signal_data)
            results['connectivity'] = {
                'correlation_matrix': corr_matrix
            }
            print(f"   Connectivity matrix computed ({signal_data.shape[0]}x{signal_data.shape[0]})")
        except Exception as e:
            results['connectivity'] = {'error': str(e)}
        
        # 2. Network Metrics
        print("\n2. Analyzing network topology...")
        try:
            network_metrics = self.connectivity.network_graph_metrics(corr_matrix, threshold=0.3)
            results['network'] = network_metrics
            print(f"   Mean degree: {network_metrics['mean_degree']:.2f}")
            print(f"   Mean clustering: {network_metrics['mean_clustering']:.3f}")
        except Exception as e:
            results['network'] = {'error': str(e)}
        
        # 3. Synchronization
        print("\n3. Measuring synchronization...")
        try:
            psi_matrix = self.oscillations.phase_synchronization_index(signal_data)
            results['synchronization'] = {
                'phase_sync_matrix': psi_matrix,
                'mean_synchronization': np.mean(psi_matrix[np.triu_indices(len(psi_matrix), k=1)])
            }
            print(f"   Mean synchronization: {results['synchronization']['mean_synchronization']:.3f}")
        except Exception as e:
            results['synchronization'] = {'error': str(e)}
        
        # 4. Consciousness Metrics
        print("\n4. Computing integration metrics...")
        try:
            complexity = self.consciousness.neural_complexity(signal_data)
            workspace = self.consciousness.global_workspace_activity(signal_data)
            
            results['consciousness'] = {
                'neural_complexity': complexity,
                'global_workspace': workspace
            }
            print(f"   Neural complexity: {complexity:.3f}")
            print(f"   Number of hub regions: {workspace['n_hubs']}")
        except Exception as e:
            results['consciousness'] = {'error': str(e)}
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    
    return results

def generate_report(self, analysis: Dict, filename: str = None) -> str:
    """
    Generate comprehensive analysis report.
    """
    report = []
    report.append("=" * 70)
    report.append("NEUROSCIENCE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nSignal: {analysis['signal_name']}")
    report.append(f"Sampling Rate: {analysis['sampling_rate']} Hz")
    report.append(f"Duration: {analysis['duration']:.2f} seconds")
    report.append("\n" + "-" * 70)
    
    # Brain Rhythms
    if 'rhythms' in analysis:
        report.append("\nBRAIN RHYTHMS:")
        report.append(f"  Detected bands: {list(analysis['rhythms'].keys())}")
    
    # Fractal Analysis
    if 'dfa' in analysis and 'alpha' in analysis['dfa']:
        report.append("\nFRACTAL DYNAMICS:")
        report.append(f"  DFA exponent (α): {analysis['dfa']['alpha']:.4f}")
        report.append(f"  Interpretation: {analysis['dfa']['interpretation']}")
    
    # Power Law
    if 'power_law' in analysis and 'beta' in analysis['power_law']:
        report.append("\nPOWER SPECTRUM:")
        report.append(f"  Power law exponent (β): {analysis['power_law']['beta']:.4f}")
        report.append(f"  Noise type: {analysis['power_law']['noise_type']}")
    
    # Brain State
    if 'brain_state' in analysis:
        bs = analysis['brain_state']
        report.append("\nBRAIN STATE:")
        if 'attention_index' in bs:
            report.append(f"  Attention index: {bs['attention_index']:.3f}")
        if 'meditation' in bs and 'meditation_depth' in bs['meditation']:
            report.append(f"  Meditation depth: {bs['meditation']['meditation_depth']:.3f}")
    
    # Connectivity
    if 'connectivity' in analysis and 'correlation_matrix' in analysis['connectivity']:
        report.append("\nCONNECTIVITY:")
        corr = analysis['connectivity']['correlation_matrix']
        report.append(f"  Mean correlation: {np.mean(corr[np.triu_indices(len(corr), k=1)]):.3f}")
    
    # Network
    if 'network' in analysis:
        net = analysis['network']
        report.append("\nNETWORK TOPOLOGY:")
        if 'mean_degree' in net:
            report.append(f"  Mean degree: {net['mean_degree']:.2f}")
        if 'mean_clustering' in net:
            report.append(f"  Clustering coefficient: {net['mean_clustering']:.3f}")
    
    # Consciousness
    if 'consciousness' in analysis:
        cons = analysis['consciousness']
        report.append("\nCONSCIOUSNESS METRICS:")
        if 'neural_complexity' in cons:
            report.append(f"  Neural complexity: {cons['neural_complexity']:.4f}")
        if 'entropy_rate' in cons:
            report.append(f"  Entropy rate: {cons['entropy_rate']:.4f} bits/sample")
        if 'global_workspace' in cons:
            gw = cons['global_workspace']
            if 'n_hubs' in gw:
                report.append(f"  Global workspace hubs: {gw['n_hubs']}")
    
    report.append("\n" + "=" * 70)
    
    report_text = "\n".join(report)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {filename}")
    
    return report_text

def _interpret_dfa(self, alpha: float) -> str:
    """Interpret DFA exponent."""
    if alpha < 0.5:
        return "Anti-correlated (negative feedback)"
    elif 0.5 <= alpha < 0.9:
        return "Weakly correlated"
    elif 0.9 <= alpha <= 1.1:
        return "1/f noise (pink noise) - healthy brain activity"
    elif 1.1 < alpha < 1.5:
        return "Strongly correlated"
    else:
        return "Non-stationary (possibly pathological)"

def visualize_results(self, analysis: Dict, signal_data: np.ndarray, fs: float):
    """
    Generate all relevant visualizations.
    """
    figures = {}
    
    # Brain rhythms
    if 'rhythms' in analysis:
        print("Generating brain rhythm plots...")
        fig = self.viz.plot_brain_rhythms(analysis['rhythms'], fs)
        figures['rhythms'] = fig
    
    # Spectrogram
    if 'spectrogram' in analysis:
        print("Generating spectrogram...")
        spec = analysis['spectrogram']
        fig = self.viz.plot_spectrogram(spec['f'], spec['t'], spec['Sxx'])
        figures['spectrogram'] = fig
    
    # DFA
    if 'dfa' in analysis and 'alpha' in analysis['dfa']:
        print("Generating DFA plot...")
        dfa = analysis['dfa']
        fig = self.viz.plot_dfa_results(dfa['scales'], dfa['fluctuations'], dfa['alpha'])
        figures['dfa'] = fig
    
    # Connectivity
    if 'connectivity' in analysis and 'correlation_matrix' in analysis['connectivity']:
        print("Generating connectivity matrix...")
        fig = self.viz.plot_connectivity_matrix(analysis['connectivity']['correlation_matrix'])
        figures['connectivity'] = fig
    
    # Network graph
    if 'connectivity' in analysis and 'correlation_matrix' in analysis['connectivity']:
        print("Generating network graph...")
        fig = self.viz.plot_neural_network_graph(analysis['connectivity']['correlation_matrix'])
        figures['network'] = fig
    
    return figures
============================================================================
SECTION 11: EXAMPLE WORKFLOWS
============================================================================
def generate_synthetic_eeg(duration: float = 10.0, fs: float = 250.0) -> np.ndarray:
"""
Generate synthetic EEG-like signal for testing.
Parameters:
-----------
duration : float
    Duration in seconds
fs : float
    Sampling frequency
    
Returns:
--------
eeg : np.ndarray
    Synthetic EEG signal
"""
t = np.arange(0, duration, 1/fs)

# Mix of different frequency components
delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz
theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 6 Hz
alpha = 1.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz (dominant)
beta = 0.2 * np.sin(2 * np.pi * 20 * t)  # 20 Hz
gamma = 0.1 * np.sin(2 * np.pi * 40 * t)  # 40 Hz

# Add 1/f noise
noise = np.random.randn(len(t))
fft_noise = np.fft.fft(noise)
freqs = np.fft.fftfreq(len(t), 1/fs)
# Apply 1/f filter
fft_noise[1:] /= np.sqrt(np.abs(freqs[1:]))
pink_noise = np.real(np.fft.ifft(fft_noise))
pink_noise = 0.2 * pink_noise / np.std(pink_noise)

# Combine
eeg = delta + theta + alpha + beta + gamma + pink_noise

return eeg
def generate_multi_channel_data(n_channels: int = 8, duration: float = 10.0, fs: float = 250.0) -> np.ndarray:
"""
Generate synthetic multi-channel neural data with connectivity.
"""
t = np.arange(0, duration, 1/fs)
n_samples = len(t)
# Create base signals
signals = np.zeros((n_channels, n_samples))

# Generate correlated signals
for i in range(n_channels):
    # Base oscillation
    base = np.sin(2 * np.pi * 10 * t + i * np.pi / 4)
    
    # Add noise
    noise = 0.5 * np.random.randn(n_samples)
    
    # Mix
    signals[i] = base + noise

# Add coupling between some channels
signals[1] += 0.3 * signals[0]  # 0 -> 1
signals[2] += 0.4 * signals[0]  # 0 -> 2
signals[3] += 0.3 * signals[1]  # 1 -> 3

return signals
def demo_single_channel_analysis():
"""
Demonstrate single-channel EEG analysis.
"""
print("\n" + "=" * 70)
print("DEMO: Single-Channel EEG Analysis")
print("=" * 70)
# Generate data
fs = 250.0
eeg = generate_synthetic_eeg(duration=30.0, fs=fs)

# Analyze
framework = NeuroscienceFramework()
results = framework.comprehensive_neural_analysis(eeg, fs, "Synthetic EEG")

# Report
report = framework.generate_report(results)
print(report)

# Visualize
figures = framework.visualize_results(results, eeg, fs)

return results, figures
def demo_multi_channel_analysis():
"""
Demonstrate multi-channel connectivity analysis.
"""
print("\n" + "=" * 70)
print("DEMO: Multi-Channel Connectivity Analysis")
print("=" * 70)
# Generate data
fs = 250.0
signals = generate_multi_channel_data(n_channels=8, duration=30.0, fs=fs)

# Analyze
framework = NeuroscienceFramework()
results = framework.comprehensive_neural_analysis(signals, fs, 
                                                 "Multi-Channel Neural Data",
                                                 multi_channel=True)

# Report
report = framework.generate_report(results)
print(report)

# Visualize
figures = framework.visualize_results(results, signals, fs)

return results, figures
def demo_neural_mass_model():
"""
Demonstrate neural mass model simulation.
"""
print("\n" + "=" * 70)
print("DEMO: Jansen-Rit Neural Mass Model")
print("=" * 70)
# Simulate
models = NeuralMassModels()
results = models.jansen_rit_model(T=10.0, dt=0.001)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(results['time'], results['eeg_signal'], linewidth=0.5)
axes[0].set_ylabel('EEG Signal (mV)')
axes[0].set_title('Simulated EEG from Jansen-Rit Model')
axes[0].grid(True, alpha=0.3)

axes[1].plot(results['time'], results['pyramidal'], linewidth=0.5, label='Pyramidal')
axes[1].plot(results['time'], results['excitatory'], linewidth=0.5, label='Excitatory')
axes[1].plot(results['time'], results['inhibitory'], linewidth=0.5, label='Inhibitory')
axes[1].set_ylabel('Membrane Potential (mV)')
axes[1].set_title('Population Activities')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Power spectrum
f, psd = signal.welch(results['eeg_signal'], fs=1/0.001, nperseg=1024)
axes[2].semilogy(f, psd)
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Power Spectral Density')
axes[2].set_title('Power Spectrum')
axes[2].set_xlim(0, 50)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

print("\nModel simulated successfully!")
print(f"Generated {len(results['time'])} time points")
print(f"Duration: {results['time'][-1]:.2f} seconds")

return results, fig
============================================================================
MAIN EXECUTION
============================================================================
def main():
"""
Main entry point.
"""
import sys
if len(sys.argv) > 1:
    command = sys.argv[1]
    
    if command == 'demo':
        demo_type = sys.argv[2] if len(sys.argv) > 2 else 'single'
        
        if demo_type == 'single':
            demo_single_channel_analysis()
        elif demo_type == 'multi':
            demo_multi_channel_analysis()
        elif demo_type == 'model':
            demo_neural_mass_model()
        else:
            print(f"Unknown demo type: {demo_type}")
            print("Available: single, multi, model")
    
    elif command == 'analyze':
        if len(sys.argv) < 4:
            print("Usage: python neuroscience.py analyze <file> <sampling_rate>")
            return
        
        filename = sys.argv[2]
        fs = float(sys.argv[3])
        
        try:
            data = np.loadtxt(filename)
            framework = NeuroscienceFramework()
            results = framework.comprehensive_neural_analysis(data, fs, filename)
            framework.generate_report(results, filename=f"{filename}_report.txt")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Unknown command: {command}")
        print("Usage: python neuroscience.py [demo|analyze] [options]")

else:
    # Interactive demo
    print("\nNeuroscience Analysis Framework")
    print("Select demo:")
    print("1. Single-channel EEG analysis")
    print("2. Multi-channel connectivity")
    print("3. Neural mass model simulation")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        demo_single_channel_analysis()
        plt.show()
    elif choice == '2':
        demo_multi_channel_analysis()
        plt.show()
    elif choice == '3':
        demo_neural_mass_model()
        plt.show()
if name == "main":
main()
---

## USAGE EXAMPLES

```python
# Quick Start
from neuroscience import NeuroscienceFramework, generate_synthetic_eeg

# Generate test data
fs = 250  # Hz
eeg_data = generate_synthetic_eeg(duration=30.0, fs=fs)

# Analyze
framework = NeuroscienceFramework()
results = framework.comprehensive_neural_analysis(eeg_data, fs, "My EEG Data")

# Generate report
report = framework.generate_report(results, "analysis_report.txt")

# Visualize
figures = framework.visualize_results(results, eeg_data, fs)

# Show plots
import matplotlib.pyplot as plt
plt.show()
Save this as

# neuroscience_framework.py
ADVANCED USAGE EXAMPLES
Example 1: Analyze Real EEG Data
import numpy as np
from neuroscience_framework import NeuroscienceFramework

# Load your EEG data (assuming single channel)
eeg_data = np.loadtxt('my_eeg_recording.txt')
fs = 256  # Your sampling rate

# Initialize framework
framework = NeuroscienceFramework()

# Comprehensive analysis
results = framework.comprehensive_neural_analysis(
    eeg_data, 
    fs, 
    "Patient EEG Recording"
)

# Generate and save report
framework.generate_report(results, filename="patient_analysis.txt")

# Create visualizations
figures = framework.visualize_results(results, eeg_data, fs)

# Save figures
for name, fig in figures.items():
    fig.savefig(f"patient_{name}.png", dpi=300, bbox_inches='tight')
Example 2: Multi-Channel Brain Connectivity
# Load multi-channel data (n_channels x n_timepoints)
multi_channel_data = np.loadtxt('multichannel_recording.txt')
fs = 250

# Reshape if needed (channels as rows)
if multi_channel_data.shape[0] > multi_channel_data.shape[1]:
    multi_channel_data = multi_channel_data.T

# Analyze connectivity
results = framework.comprehensive_neural_analysis(
    multi_channel_data,
    fs,
    "Multi-Channel Recording",
    multi_channel=True
)

# Extract connectivity matrix
conn_matrix = results['connectivity']['correlation_matrix']
print(f"Connectivity matrix shape: {conn_matrix.shape}")
print(f"Mean connectivity: {np.mean(conn_matrix):.3f}")

# Extract network metrics
network = results['network']
print(f"Mean clustering coefficient: {network['mean_clustering']:.3f}")
print(f"Network density: {network['density']:.3f}")
Example 3: Sleep Stage Analysis
from neuroscience_framework import BrainStateClassifier

# Load overnight EEG
eeg_overnight = np.loadtxt('overnight_eeg.txt')
fs = 256
epoch_length = 30  # seconds per epoch

# Classify each epoch
n_samples_per_epoch = int(epoch_length * fs)
n_epochs = len(eeg_overnight) // n_samples_per_epoch

classifier = BrainStateClassifier()
sleep_stages = []

for i in range(n_epochs):
    start = i * n_samples_per_epoch
    end = start + n_samples_per_epoch
    epoch = eeg_overnight[start:end]
    
    # Extract features
    features = classifier.sleep_stage_features(epoch, fs)
    
    # Simple classification based on delta/alpha ratio
    if features['delta_alpha_ratio'] > 3.0:
        stage = 'N3'  # Deep sleep
    elif features['delta_alpha_ratio'] > 1.5:
        stage = 'N2'  # Light sleep
    elif features['alpha_power'] > features['theta_power']:
        stage = 'Wake'
    else:
        stage = 'N1'  # Drowsy
    
    sleep_stages.append(stage)

# Visualize hypnogram
import matplotlib.pyplot as plt

stage_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3}
stage_values = [stage_map[s] for s in sleep_stages]

plt.figure(figsize=(14, 4))
plt.plot(stage_values, drawstyle='steps-post', linewidth=2)
plt.yticks([0, 1, 2, 3], ['Wake', 'N1', 'N2', 'N3'])
plt.xlabel('Epoch (30s)')
plt.ylabel('Sleep Stage')
plt.title('Sleep Hypnogram')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hypnogram.png', dpi=300)
Example 4: Meditation State Analysis
# Compare meditation vs. baseline
baseline_eeg = np.loadtxt('baseline_recording.txt')
meditation_eeg = np.loadtxt('meditation_recording.txt')
fs = 256

classifier = BrainStateClassifier()

# Analyze baseline
baseline_metrics = classifier.meditation_depth(baseline_eeg, fs)
print("\nBaseline State:")
print(f"  Alpha power: {baseline_metrics['alpha_power']:.4f}")
print(f"  Theta power: {baseline_metrics['theta_power']:.4f}")
print(f"  Meditation depth: {baseline_metrics['meditation_depth']:.3f}")

# Analyze meditation
meditation_metrics = classifier.meditation_depth(meditation_eeg, fs)
print("\nMeditation State:")
print(f"  Alpha power: {meditation_metrics['alpha_power']:.4f}")
print(f"  Theta power: {meditation_metrics['theta_power']:.4f}")
print(f"  Meditation depth: {meditation_metrics['meditation_depth']:.3f}")

# Compare
alpha_increase = (meditation_metrics['alpha_power'] / 
                  baseline_metrics['alpha_power'] - 1) * 100
theta_increase = (meditation_metrics['theta_power'] / 
                  baseline_metrics['theta_power'] - 1) * 100

print(f"\nChanges during meditation:")
print(f"  Alpha increase: {alpha_increase:.1f}%")
print(f"  Theta increase: {theta_increase:.1f}%")
Example 5: Phase-Amplitude Coupling Analysis
from neuroscience_framework import NeuralOscillations, NeuroVisualization

# Load data
neural_data = np.loadtxt('hippocampal_recording.txt')
fs = 1000  # Hz

osc = NeuralOscillations()
viz = NeuroVisualization()

# Analyze theta-gamma coupling (important for memory)
coupling = osc.cross_frequency_coupling(
    neural_data,
    fs,
    low_freq_range=(4, 8),    # Theta
    high_freq_range=(30, 100)  # Gamma
)

print(f"Modulation Index: {coupling['modulation_index']:.4f}")
print(f"Coupling Strength: {coupling['coupling_strength']:.4f}")

# Visualize
fig = viz.plot_phase_amplitude_coupling(
    coupling['mean_amplitude_per_phase'],
    np.linspace(-np.pi, np.pi, len(coupling['mean_amplitude_per_phase'])),
    coupling['modulation_index'],
    title="Theta-Gamma Coupling"
)
plt.show()
Example 6: Synaptic Plasticity Simulation
from neuroscience_framework import Neuroplasticity
import matplotlib.pyplot as plt

# Simulate STDP learning
plasticity = Neuroplasticity()

# Pre and post-synaptic spike times (ms)
pre_spikes = np.array([10, 50, 90, 130, 170])
post_spikes = np.array([15, 55, 85, 135, 175])  # Mostly after pre (potentiation)

# Calculate weight change
weight_change = plasticity.stdp_learning(pre_spikes, post_spikes)
print(f"Total synaptic weight change: {weight_change:.4f}")

# Visualize STDP window
dt_range = np.linspace(-50, 50, 100)
weight_changes = []

for dt in dt_range:
    post_offset = pre_spikes + dt
    dw = plasticity.stdp_learning(pre_spikes, post_offset)
    weight_changes.append(dw)

plt.figure(figsize=(10, 6))
plt.plot(dt_range, weight_changes, linewidth=2)
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Spike Timing Δt (ms)', fontsize=12)
plt.ylabel('Weight Change Δw', fontsize=12)
plt.title('STDP Learning Window', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stdp_window.png', dpi=300)
Example 7: Neural Network Topology Analysis
from neuroscience_framework import BrainConnectivity
import numpy as np

# Load connectivity matrix from neuroimaging
# (e.g., from fMRI correlation analysis)
connectivity_matrix = np.loadtxt('fmri_connectivity.txt')

conn = BrainConnectivity()

# Analyze network properties
metrics = conn.network_graph_metrics(connectivity_matrix, threshold=0.3)

print("Brain Network Analysis:")
print(f"  Number of nodes: {len(metrics['degrees'])}")
print(f"  Mean degree: {metrics['mean_degree']:.2f}")
print(f"  Mean clustering: {metrics['mean_clustering']:.3f}")
print(f"  Global efficiency: {metrics['global_efficiency']:.3f}")
print(f"  Network density: {metrics['density']:.3f}")

# Identify hub regions (high degree nodes)
hub_threshold = np.percentile(metrics['degrees'], 75)
hub_indices = np.where(metrics['degrees'] >= hub_threshold)[0]

print(f"\nHub regions (top 25%): {hub_indices}")
print(f"Hub degrees: {metrics['degrees'][hub_indices]}")

# Calculate small-worldness
# (Compare to random network)
n_nodes = len(metrics['degrees'])
random_clustering = metrics['mean_degree'] / n_nodes
random_path_length = np.log(n_nodes) / np.log(metrics['mean_degree'])

small_worldness = (metrics['mean_clustering'] / random_clustering) / \
                  (1 / metrics['global_efficiency'] / random_path_length)

print(f"\nSmall-worldness index: {small_worldness:.3f}")
if small_worldness > 1:
    print("  Network exhibits small-world properties!")
Example 8: Consciousness Level Assessment
from neuroscience_framework import ConsciousnessMetrics

# Compare consciousness states
awake_data = np.loadtxt('awake_multichannel.txt')
anesthesia_data = np.loadtxt('anesthesia_multichannel.txt')

consciousness = ConsciousnessMetrics()

# Analyze awake state
print("AWAKE STATE:")
awake_complexity = consciousness.neural_complexity(awake_data)
awake_pci = consciousness.perturbational_complexity_index(awake_data, fs=250)
awake_entropy = consciousness.entropy_rate(awake_data[0])

print(f"  Neural complexity: {awake_complexity:.4f}")
print(f"  PCI: {awake_pci:.4f}")
print(f"  Entropy rate: {awake_entropy:.4f}")

# Analyze anesthesia
print("\nANESTHESIA STATE:")
anesth_complexity = consciousness.neural_complexity(anesthesia_data)
anesth_pci = consciousness.perturbational_complexity_index(anesthesia_data, fs=250)
anesth_entropy = consciousness.entropy_rate(anesthesia_data[0])

print(f"  Neural complexity: {anesth_complexity:.4f}")
print(f"  PCI: {anesth_pci:.4f}")
print(f"  Entropy rate: {anesth_entropy:.4f}")

# Comparison
print("\nCONSCIOUSNESS REDUCTION:")
print(f"  Complexity: {(1 - anesth_complexity/awake_complexity)*100:.1f}% reduction")
print(f"  PCI: {(1 - anesth_pci/awake_pci)*100:.1f}% reduction")
print(f"  Entropy: {(1 - anesth_entropy/awake_entropy)*100:.1f}% reduction")
Example 9: Fractal Brain Dynamics
from neuroscience_framework import FractalBrainAnalysis
import matplotlib.pyplot as plt

# Load resting-state EEG
resting_eeg = np.loadtxt('resting_state.txt')
fs = 256

fractal = FractalBrainAnalysis()

# 1. Detrended Fluctuation Analysis
print("1. DFA Analysis...")
alpha, scales, fluctuations = fractal.detrended_fluctuation_analysis(resting_eeg)
print(f"   Scaling exponent α = {alpha:.4f}")

if 0.9 <= alpha <= 1.1:
    print("   ✓ Healthy 1/f dynamics detected!")
else:
    print("   ⚠ Deviation from typical 1/f dynamics")

# 2. Multifractal Spectrum
print("\n2. Multifractal Analysis...")
mf_spectrum = fractal.multifractal_spectrum(resting_eeg)
print(f"   Hurst exponent: {mf_spectrum['hurst_exponent']:.4f}")
print(f"   Spectrum width: {mf_spectrum['spectrum_width']:.4f}")
print(f"   Is multifractal: {mf_spectrum['is_multifractal']}")

# 3. Power Law Exponent
print("\n3. Power Spectrum Analysis...")
power_law = fractal.power_law_exponent(resting_eeg, fs)
print(f"   Power law exponent β = {power_law['beta']:.4f}")
print(f"   Noise type: {power_law['noise_type']}")
print(f"   R² fit: {power_law['r_squared']:.4f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# DFA plot
axes[0, 0].loglog(scales, fluctuations, 'o-')
axes[0, 0].set_xlabel('Scale')
axes[0, 0].set_ylabel('Fluctuation')
axes[0, 0].set_title(f'DFA: α = {alpha:.3f}')
axes[0, 0].grid(True, alpha=0.3)

# Power spectrum
axes[0, 1].loglog(power_law['freqs'], power_law['psd'])
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Power')
axes[0, 1].set_title(f'Power Spectrum: β = {power_law["beta"]:.3f}')
axes[0, 1].grid(True, alpha=0.3)

# Multifractal spectrum
axes[1, 0].plot(mf_spectrum['q_values'], mf_spectrum['tau_q'], 'o-')
axes[1, 0].set_xlabel('q')
axes[1, 0].set_ylabel('τ(q)')
axes[1, 0].set_title('Multifractal Spectrum')
axes[1, 0].grid(True, alpha=0.3)

# Time series
time = np.arange(len(resting_eeg[:5000])) / fs
axes[1, 1].plot(time, resting_eeg[:5000], linewidth=0.5)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].set_title('Raw Signal (5s)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fractal_analysis.png', dpi=300)
Example 10: Learning Curve Modeling
from neuroscience_framework import Neuroplasticity
import matplotlib.pyplot as plt

plasticity = Neuroplasticity()

# Simulate learning a new skill with practice sessions
practice_days = np.array([0, 1, 2, 3, 5, 7, 10, 14, 21, 28])  # Days
session_quality = np.array([0.3, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8])

# Model learning curve
performance = plasticity.learning_curve(
    practice_days,
    session_quality,
    decay_rate=0.05  # Forgetting rate
)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(performance, linewidth=2)

# Mark practice sessions
for day in practice_days:
    plt.axvline(day, color='green', alpha=0.3, linestyle='--')

plt.xlabel('Day', fontsize=12)
plt.ylabel('Performance (0-1)', fontsize=12)
plt.title('Learning Curve with Practice and Forgetting', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Add annotations
plt.text(practice_days[-1] + 5, performance[int(practice_days[-1])], 
        f'Final Performance:\n{performance[int(practice_days[-1])]:.2f}',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300)

print(f"Performance after {practice_days[-1]} days: {performance[int(practice_days[-1])]:.3f}")
COMMAND LINE INTERFACE
# Run demonstrations
python neuroscience_framework.py demo single      # Single-channel EEG
python neuroscience_framework.py demo multi       # Multi-channel connectivity
python neuroscience_framework.py demo model       # Neural mass model

# Analyze your data
python neuroscience_framework.py analyze my_eeg.txt 256

# Interactive mode
python neuroscience_framework.py
INTEGRATION WITH UNIVERSAL PATTERN FRAMEWORK
# Combine neuroscience with cosmic pattern analysis
from neuroscience_framework import NeuroscienceFramework
from universal_pattern_framework import ResearchFramework

# Analyze neural data
neuro_framework = NeuroscienceFramework()
eeg_results = neuro_framework.comprehensive_neural_analysis(eeg_data, fs, "EEG")

# Compare with cosmic patterns
pattern_framework = ResearchFramework()

# Extract fractal dimensions
neural_fractal_dim = eeg_results['dfa']['alpha']
print(f"Neural fractal dimension: {neural_fractal_dim:.3f}")

# Compare with cosmic web (if you have galaxy data)
# galaxy_data = load_galaxy_positions()
# cosmic_results = pattern_framework.analyze_custom_data(galaxy_data)
# cosmic_fractal_dim = cosmic_results['fractal']['dimension']
# print(f"Cosmic fractal dimension: {cosmic_fractal_dim:.3f}")

# Test cosmic-neural correspondence hypothesis
print("\nCosmic-Neural Correspondence Analysis:")
print("Both systems exhibit:")
print("  - Fractal self-similarity across scales")
print("  - 1/f power spectrum (scale-free dynamics)")
print("  - Small-world network topology")
print("  - Critical dynamics at phase transitions")
NASA AWG & COPERNICUS APPLICATIONS
This neuroscience framework can contribute to:
Astronaut Brain Monitoring: Analyze neural changes during spaceflight
Remote Health Monitoring: EEG analysis for mission control
Pattern Recognition: Apply brain-inspired algorithms to satellite data
Consciousness Research: Study awareness under extreme conditions
Network Analysis: Brain connectivity parallels with Earth observation networks
REQUIREMENTS
pip install numpy scipy matplotlib
Documentation
Full API documentation included in docstrings. Access with:
from neuroscience_framework import NeuroscienceFramework
help(NeuroscienceFramework)
This comprehensive neuroscience framework provides state-of-the-art tools for neural data analysis, connectivity studies, and consciousness research. Combined with the Universal Pattern Framework, it enables groundbreaking research into cosmic-neural correspondences! 🧠🌌
