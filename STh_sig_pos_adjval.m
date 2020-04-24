function [res] = STh_sig_pos_adjval(x,thi,taui)
%smooth sigmoid based hard thresholding
[M,N,T] = size(x);
res = zeros(M,N,T,'single');

for ii = 1:T
vec_xii = vec(x(:,:,ii));
sortvec = sort(vec_xii, 'descend');
L = length(sortvec);
st = sortvec(round(0.99*L)); %min
en = sortvec(round(0.01*L)); %en = prctile(vec,99);
th = st + (en-st)*thi; %thi is restricted to be between 0 and 1

if th<1e-14
    th = 0;
end

if th ~=0
tau = taui/th;
else
tau = taui;
end

res(:,:,ii) = relu(x(:,:,ii))./(1+exp(-tau*(abs(x(:,:,ii))-th)));
% for i = 1:M
%     for j = 1:N
%     res(i,j,ii) = relu(x(i,j,ii))/(1+exp(-tau*(abs(x(i,j,ii))-th)));
%     end
% end

end

end

