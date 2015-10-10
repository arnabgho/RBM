function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    % error('not yet implemented');
    visible_state=sample_bernoulli(visible_data)

    hidden_data=visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_data_state=sample_bernoulli(hidden_data)

    del_rbm_w=hidden_data_state*visible_state';

    visible_data_reconstruct=hidden_state_to_visible_probabilities(rbm_w, hidden_data_state);
    visible_data_reconstruct_state=sample_bernoulli(visible_data_reconstruct)

    hidden_data_reconstruct=visible_state_to_hidden_probabilities(rbm_w, visible_data_reconstruct_state);
    % hidden_data_reconstruct_state=sample_bernoulli(hidden_data_reconstruct)

    % del_rbm_w-=hidden_data_reconstruct_state*visible_data_reconstruct_state'; 
     del_rbm_w-=hidden_data_reconstruct*visible_data_reconstruct_state'; 
    [n m]=size(visible_data)
    ret=(rbm_w+del_rbm_w)/m;
end
