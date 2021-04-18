function im = distort_image(im,prop)
if (prop<0 || prop>1)
    disp('Out-of-bound proportion: going to default 0.05')
    prop = 0.05; %Default
end
indx = randperm(length(im(:)));
todist = indx(1:round(length(indx)*prop));
im(todist) = -im(todist);
end