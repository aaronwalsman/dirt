def train(
    key,
    ga_params,
):
    
    if parameter_type == 'GPU_MEMORY':
        initialize_shared_parameters = lambda key, n : None
        initialize_shared_parameters = vmap(model, ...)
    elif parameter_type == 'WHATEVER ELSE':
        ...
    
    # wrap the model in a big_pop
    model, initialize_parameters, mutate_parameters = big_pop(
        parallel_model,
        initialize_shared_parameters,
        initialize_player_parameters,
        parameters_to_weights,
        mutate_model_setitem,
        mutate_model_getitem,
    )
    
    initialize_ga, step_ga = ga(
        ga_params,
        initialize_model,
    )
    
    '''
    So here's where I got with this:
    The model forward function just needs to be able to take extra dimension
    in the weights.  It's such a small thing, just do that.  The shapes
    should be (B1, B2, B3, ..., M1, M2, M3, ..., H, W, C) for data
    and (M1, M2, M3, ... H, W, C) for the model parameters.
    
    Then the initialization script should just take a model_shape
    (parallel_shape?) parameter that defaults to an empty tuple.
    
    Then mutation.  Also takes (M1, M2, M3, ... H, W, C).  And we're fine.
    That's it.  No vmap, no bullshit.
    
    So I think the next thing is to actually build out some model code.
    
    Then figure out how to reconfigure big_pop based on that.
    
    Then get GA up and running with a very simple MLP model.
    
    Ok, so here's what a neural network model looks like:
    
    Each model (layer, network, whatever) is a function that produces:
        initailize(key, shape=()) : a random function that initializes the
            parameters of that model.  Parameters can be whatever shape or
            pytree structure you want.
        model(key, x, parameters) : a potentially random function that takes
            x and the parameters and generates a new x.  The input and output
            x values can have whatever shape/pytree structure you want.
    
    Now we need mutators.  These are random functions that take in a set of
        model parameters and generate a new set of perturbed model parameters.
    
    The last thing we need is a genotype to phenotype mapping function.  Could
        be randomized.  Probably want the randomization to happen once for the
        lifetime of the organism though.
    
    Then big_pop comes in and provides a tool to build wrappers on top of
        existing layers.  It takes the genotype to phenotype matching function
        and makes a chunked pipeline that will instantiate one set of model
        weights at a time.
    
    Thinking about this further though, I think the functions of big_pop
        should be split out.  The only thing that big_pop is really needed
        for is doing the chunked instantiated for loop.  So really, that could
        just be a single wrapper on the forward pass.
    
    Likewise the genotype to phenotype mapper could be a wrapper on the
        original model's initialization and forward functions.
    
    Finally what about the mutators?  I guess they are just like optimizers
        and have their own separate parameters.  You use different ones
        depending on the data type of the model you are working with.
    
    The last thing to think about is this whole shared_parameters,
        player_parameters distinction.  This is necessary for the chunker,
        so that it knows what parameters to split up into chunks and what
        should get passed to everything.  I'm tempted to just force all
        initializers to return a shared_parameters, player_parameters tuple
        so that you could use the chunking wrapper around anything you wanted.
        I already did some shoe-horning and forced all forward passes to take
        a key even though many of them don't need it.  Probably fine, and
        makes the rest of it pretty trivial.
    
    Ok, so now let's talk about genotype to phenotype mapping.  As we said
        before this is just a set of wrappers around the model.  For a mapping
        that uses integers to generate weights, we need to know the shapes of
        the weights we are dealing with.  One temptation is to put the weights
        and bias of linear/conv layers into dictionaries with "weight" and
        "bias" keys, so they can be easily identified by a mapper.
        
        The idea is that you would pass the original initialize/model functions
        into a map maker and it would initialize a single copy of the weights,
        then scan it to see what shape they are and make the appropriate
        mapping functions.
        
        An alternative is to just make a replacement for linear/conv layers and
        skip the wrapper setup.  The problem with that though, is that then
        you need to slot that new thing into all the low-level MLPs and conv
        stacks and so on, which is a bit of a pain.  It also means if there is
        suddenly some other linear-ish layer you want to build you need to
        make another replacement for it.  Yeah, seems... unsightly.
        
        So I kind of like the wrapper I guess, but it does add a little bit
        more structure to the parameter data.  But the advantage is that you
        can just wrap the entire network after the fact, and can keep the low
        level code very simple.
        
        Oh shoot though, here's the issue.  If there are per-model parameters
        that should not be replaced by indices, how does the wrapper initialize
        a large array of them without running the original initializer and
        blowing up memory.  So maybe we go one step more radical and we say
        that every per-player parameter has to be replaced in these things?
        
        And then if you have per-player parameters that you don't want
        replaced, you can just move the wrapper inward somewhat.
        
        Ok, yet another question: how do we set the magnitude of the library
        weights?  We can do some back-of-the-napkin based on how many of them
        we are summing in order to get the result to be a kaiming-scaled
        normal distribution.  I guess what we really want is not whatever
        magnitude we would have at initialization, but the magnitude that a
        supervised learning run would eventually have after convergence?
        We'll have to play with it.  What about bias though?
        
        Another option would be to allow the magnitude to be another
        parameter that can move around, and just optimze that too.
        
        Lots of design decisions and questions here.
        
        Structurally though, how would I change these individually if I wanted
        to or needed to if it's just a wrapper that gets to see a bunch of
        tensor shapes at a particular initialization point?  OH!  Just use the
        magnitude of the initialized thing!  Yeah this is the way.  Then you
        can control things individually based on how they are initialized
        earlier.
        
        This still relies on doing this everywhere in the player_parameters,
        but does let me get away from labelling weights and biases.  It does
        mean that I can't initialize biases with zeros though... which is fine.
        But basically it's like I want to give it a trained model, or have
        some understanding of what the magnitude of various things will be at
        the end of training.
        
        Ok, ok, ok ok, one more hiccup.  How do I initialize the shared
        parameters???  Again if these come from one function, how do I get
        them without doing a full run.  Oh, wait, this is easier, just do
        an initialization of a single model, and use the shared parameters of
        that.  Move the computation inside the initialization function.
        
        So I have this sketched out, except for the mutation.  The rough
        psuedocode for building the model would be:
        
        initialize, model = mlp(...)
        initialize, model = mutation_library(initialize, model, ...)
        model = model_chunk(model, ...)
        
        parameters = initailize(initialize_key)
        
        x = model(model_key, x, parameters)
        
        Seems good?
    
    One more note on building the whole GA.  We need mutation to be able to
        accept partial lists of player_parameters.  This is fine for all our
        cases I think.  The other thing I need is to be able to copy player
        indices from one generation to another.  Pretty easy, but needs to be
        done.  I guess I already have it with tree_getitem and tree_setitem.
    '''
