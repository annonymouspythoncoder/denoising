#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def SGDenoise( #Returns the denoised audio (without some of the original noise) 
    Audio_clip, #x
    num_freq_chan=1,         # of frequency channels that will receive smoothed mask. #x
    num_time_chan=4,         # of time channels that will receive smoothed mask.
    num_fft_frames=1024,     # of number audio of frames between STFT columns. #x
    win_length=1024,         # applied to window length used for each audio frame, matching num_fft_frames numbers
    hop_length=128,          # of audio frames that are skipped (hopped) between STFT columns. #x
    max_amp_over_noise=1.5,  # Maximum allowed loudness over the mean noise amplitude 
    denoise_factor=1.0      # How much should noise be decreased (1 = all, 0 = none)
):
      
    # Calculates noise statistics. 
    noise_stft = librosa.stft(y=audio_clip, n_fft=num_fft_frames, hop_length=hop_length, win_length=win_length) #Brings audio sample as noise from time series to vectorized spectrogram through short Fourier Transform   
    noise_stft_amp = librosa.core.amplitude_to_db(np.abs(noise_stft), ref=1.0, amin=1e-20, top_db=80.0) #x #converts frequencies to to amplitude in dB
    mean_freq_noise = np.mean(noise_stft_amp, axis=1) #computes average amplitude of the noise signal array
    std_freq_noise = np.std(noise_stft_amp, axis=1)  #computes standard deviation of amplitude of the noise signal array
    noise_profile = (mean_freq_noise + std_freq_noise * max_amp_over_noise) #generates noise profile
    
    # Calculates audio statistics. 
    audio_stft = librosa.stft(y=audio_clip, n_fft=num_fft_frames, hop_length=hop_length, win_length=win_length) #Brings audio sample from time series to vectorized spectrogram through short Fourier Transform
    audio_stft_amp =  librosa.core.amplitude_to_db(np.abs(audio_stft), ref=1.0, amin=1e-20, top_db=80.0) #converts frequencies to to amplitude in dB
    audio_profile = np.repeat(    # Audio profile takes into account the noise profile 
                                np.reshape(noise_profile, [1, len(mean_freq_noise)]),
                                np.shape(audio_stft_amp)[1],
                                axis=0,
    ).T
     
    ############################################
    # Calculates initial denoising mask to apply to audio
    mask_gain_amp = np.min(librosa.core.amplitude_to_db(np.abs(audio_stft), ref=1.0, amin=1e-20, top_db=80.0)) #x ### Gain is computed as the minimum amplitude element in absolute values over fourier transformed signal  
    
    # Creates a filter by making the initial gain more linear (array is evenly spaced), smoothing out mask 
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(-1, 1, num_freq_chan + 2, endpoint=False), #x
                np.linspace(1, -1, num_freq_chan + 2), #x
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(-1, 1, num_time_chan + 2, endpoint=False), #x
                np.linspace(1, -1, num_time_chan + 2), #x
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter * (smoothing_filter / np.sum(smoothing_filter)) #x
    ############################################  

    #defines, at the audio bin level, where the mask will be applied
    audio_mask = audio_stft_amp < audio_profile    
    # Applies smoothing filter to mask
    audio_mask = scipy.signal.fftconvolve(audio_mask, smoothing_filter, mode="same")
    audio_mask = audio_mask * denoise_factor        
    
    # Masks the audio
    audio_stft_amp_masked = (
        audio_stft_amp * (1 - audio_mask)
        + np.ones(np.shape(mask_gain_amp)) * mask_gain_amp * audio_mask
    ) 
    audio_imag_masked = np.imag(audio_stft) * (1 - audio_mask)
    audio_stft_amp = (librosa.core.db_to_amplitude(audio_stft_amp_masked, ref=1.0) * np.sign(audio_stft)) + (
        1j * audio_imag_masked
    )
    ############################################
    # performs inverse fourier transform to bring audio from vectorized spectrogram to time series domain
    denoised_audio = librosa.istft(stft_matrix=audio_stft_amp, hop_length=hop_length, win_length=win_length)
    denoised_spec = librosa.core.amplitude_to_db(np.abs(librosa.stft(y=denoised_audio, n_fft=num_fft_frames, hop_length=hop_length, win_length=win_length)), ref=1.0, amin=1e-20, top_db=80.0) 
    
    return denoised_audio

