function x_outs = LSPARCOM_VAR_NET2(input, NET_WEIGHTS, numfolds)
%<><><><><><<><>><><><>FORWARD PASS<><><><><><><><><><><>
Lfv = conv2D_layer(NET_WEIGHTS.convA_field_0, input, 'same');
xw = Lfv;

for k = 0:numfolds-1 
   x_thresh = STh_sig_pos_adjval(xw, NET_WEIGHTS.(['x_thresh_', num2str(k),'_field_0']),NET_WEIGHTS.(['x_thresh_', num2str(k),'_field_1']));
   x_thresh_P1 = conv2D_layer(NET_WEIGHTS.(['convM_', num2str(k),'_field_0']), x_thresh, 'same');
   xw = Lfv - x_thresh_P1 + x_thresh;
end

x_out = STh_sig_pos_adjval(xw, NET_WEIGHTS.prox_out_field_0 ,NET_WEIGHTS.prox_out_field_1);
x_outs = NET_WEIGHTS.out_scale_field_0*x_out;
end
