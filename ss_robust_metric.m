function [  ] = compare_all(inpd, vss)

	%resfile = strcat(inpd, resfile);

    for test = 1:length(vss)
        gt_p  = sprintf('%s/%s.avi', inpd, vss{test}{2});
        cur_p = sprintf('%s/%s.avi', inpd, vss{test}{1});
        circles_p = sprintf('%s/circles_%s.avs/circles_%s.avi', inpd, vss{test}{2}, vss{test}{2});

        if( ~exist(gt_p,'file') || ~exist(cur_p,'file') )
            fprintf('%s vs %s Not found\r\n',  vss{test}{1},  vss{test}{2});
        else
            [psnr ss AUC]= compare_v(cur_p, gt_p, circles_p, vss{test}{3}, vss{test}{4});
%             fp = fopen(resfile,'a');
%                 fprintf(fp,'%f %f %f\n', psnr, ss, AUC);
%             fclose(fp);
        end
    end
end

function [psnr, ss, AUC] = compare_v(src_avi, gt_avi, circles_avi, resfile, x)

    cur     = VideoReader(src_avi);
    ref     = VideoReader(gt_avi);
    if~strcmp(circles_avi, '')
        circles = VideoReader(circles_avi);
    end
    
	if (isempty(x))
		x = opt(sample(cur, ref, 400), resfile);
	end
	
    fn = min(cur.NumberOfFrames, ref.NumberOfFrames);

    psnr = zeros(fn, 1);
    ss   = zeros(fn, 1);
    AUC  = zeros(fn, 1); 

    for i = 1 : fn
        d.cur     = applyt(xrgb2gray(im2double(read(cur, i))), x);
        d.ref     = xrgb2gray(im2double(read(ref,i)));
    
        if ~strcmp(circles_avi, '')
            d.circles = xrgb2gray(im2double(read(circles,i)));
            [AUC(i), ~] = sm_roc(d.cur, d.circles);
        end
         
        ssd         = mean( (d.ref(:) - d.cur(:)).^2 );
        psnr(i)     = 10*log(1/(ssd)) / log(10);
        ss(i)       = sum(min( d.cur(:) / sum(d.cur(:)), d.ref(:) / sum(d.ref(:)) ));

        xlog = fopen(resfile, 'a');	
        fprintf(xlog, '%d / %d psnr = %f ss = %f AUC = %f\n', i, fn, psnr(i), ss(i), AUC(i));
        fclose(xlog);
    end

    psnr = (sum(psnr(isfinite(psnr))) + 100.0 * sum(isinf(psnr))) / size(psnr(:), 1);
    ss   = sum(ss(isfinite(ss))) / size(ss(:), 1);
    AUC  = mean(AUC(isfinite(AUC)));

    xlog = fopen(resfile, 'a');
    fprintf (xlog, '---------------FINAL---------------\n');
    fprintf (xlog, 'psnr = %f ss = %f AUC = %f\n\n\n', psnr, ss, AUC);
    fclose(xlog);
end

function [res] = gt_dif(x, d)
    r = applyt(d.cur, x);
    %res = mean( (d.ref(:)-r(:)).^2 );
	
	nr = reshape(r(:), size(r,2)*size(r,1), size(r,3));
	nr = nr ./ repmat(sum(nr, 1), [size(nr, 1), 1]);
	nr(isnan(nr)) = 0;
    
	nd = reshape(d.ref(:), size(d.ref,2)*size(d.ref,1), size(d.ref,3));
	nd = nd ./ repmat(sum(nd, 1), [size(nd, 1), 1]);
	nd(isnan(nd)) = 0;
    
	res = 100  - mean(sum(min(nd, nr), 1));	
end

function [r] = applyt(A, x)
    global ABSFDefault;
    if (sum(size(ABSFDefault) == [1080 1920]) ~= 2)
       xx  = -1920/2:1920/2 - 1; sigmaX = 1920/7;
       yy  = -1080/2:1080/2 - 1; sigmaY = 1080/7;
       GX =  exp(-xx.^2 / (2*sigmaX^2));
       GY =  exp(-yy.^2 / (2*sigmaY^2));
       ABSFDefault = GY' * GX;
       
       ABSFDefault = levels(ABSFDefault, [0.0011 0.9974 0.00 1 0.8598]);
    end

    r = levels(A, x(2:6));
    
    r = r * (1.0 - x(1)) + repmat(ABSFDefault, [1 1 size(r,3)]) * x(1);    
end

function [r] = levels(A, x)
    i_b = min(1, max(0, x(1)));
    i_w = max(i_b, min(1, max(0,x(2))));
    o_b = min(1, max(0, x(3)));
    o_w = max(o_b, min(1, max(0,x(4))));
    g   = max(0, x(5));

    r = min(max(A - i_b, 0) / (i_w - i_b), 1);
    r = r.^g * (o_w - o_b) + o_b;
end

function [data] = sample(cur, ref, step)
  
    fn = min(ref.NumberOfFrames, cur.NumberOfFrames);
    for i = 1:step:fn
       data.ref(:, :, floor((i - 1) / step) + 1) = xrgb2gray(im2double(read(ref, i)));
       data.cur(:, :, floor((i - 1) / step) + 1) = xrgb2gray(im2double(read(cur, i)));
       
       fprintf('sampling %d / %d\n', i, fn);
    end
    
end

function [ x ] = opt(data, resfile)

    x0 = [0.5  0.0  0.5  0.0  0.5  1.0];
    lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    ub = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0];

    opt = optimset('Display', 'iter', 'UseParallel', 'always');
    x = fmincon(@gt_dif, x0, [], [], [], [], lb, ub, [], opt, data);
    xlog = fopen(resfile, 'a');
    fprintf(xlog, strcat('CP=%.2f  IB=%.2f  IW=%.2f  OB=%.2f  OW=%.2f  AL=%.2f\n',...
                         '....................................................\n'),...
            x(1), x(2), x(3), x(4), x(5), x(6));
    fclose(xlog);
end

function [area, roc] = sm_roc(cur, ref)
    cur = cur(:);
    ref = ref(:);
    
	ref = ref > 0;
	roc = zeros(255,2);
	
	n = 255;
	st = 1;

	all_p = sum(ref(:));
	all_n = length(ref(:)) - all_p;

	p_i = ref(:) == true;
	%n_i = find( ref(:) == false);
    rng = (0:(n)) / n; 
    c_h =   flipdim(    cumsum( flipdim(   histc( cur , rng )       ,1    ) )   ,1);
    c_h_p = flipdim(    cumsum( flipdim(   histc( cur(p_i) , rng )  ,1    ) )   ,1);
    
    

	for thr = 1:n+1	
		true_p = c_h_p(thr);
		%false_p = sum( bin(n_i) );
		false_p = c_h(thr) - true_p;
		%false_n = sum( ~bin(:) & ref(:) );
		%true_n = length(bin(:)) - true_p - false_p - false_n;
		
		tpr = true_p / (all_p);
		fpr = false_p / (all_n);
		
		roc(st, 2) = tpr;
		roc(st, 1) = fpr;
		st = st + 1;
	end
	roc  = sortrows(roc,1);
    %plot(roc(:,1),roc(:,2));
	area =   trapz(roc(:,1),roc(:,2));

end


function res = xrgb2gray(img)
    if ((sum(size(size(img))) == 4) & size(img,3) == 3)
        res = rgb2gray(img);
    else
        res = img;
    end
end