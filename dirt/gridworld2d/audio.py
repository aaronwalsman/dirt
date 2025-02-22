def sample_audio(
    audio_source_locations, audio_source_signals, sampled_locations
):
    # TODO: Vincent explore ideas here
    
    '''
    Idea 1 (Convolution): Starting from a static model with N^2 attenuation, then
    use convolution kernal and newly-defined decay factors to update:
    S^(t+1) = \gamma*K*S^t + S_{audio source}^{t+1}

    Idea 2 (Field): 
    Even more naively, we can simply apply advection, 
    where we push the sound intensity field across the grid according to a velocity field
    Apply the idea of ballistic propagation model


    Add-on Idea:
    -  We could try stopping the sound when the distance is large enough to save the diffusion
    -  Using propagation delay for dynamic sound diffusion

    '''
    

    



    return sampled_signals
