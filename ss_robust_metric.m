function [ ss,per_frame_ss ] = ss_robust_metric(src_avi, gt_avi,sampling_density,optimization_iterations)

    cur     = VideoReader(src_avi);
    ref     = VideoReader(gt_avi);

    
    if( ~exist('sampling_density', 'var' ) )
        sampling_density = 400;
    end
    
    if( ~exist('optimization_iterations', 'var' ) )
        optimization_iterations = 100;
    end
        
    
    center_prior = get_center_prior(gt_avi,ref,sampling_density,optimization_iterations);
    
    fprintf('==Sampling every %d-th frame to choose best settings==\n', sampling_density);
    samples = sample(cur, ref, sampling_density);
    samples.center_prior = center_prior;
    fprintf('==Searching for best parameters==\n');
    
    min_fval = +Inf;
    
    for i = 1:optimization_iterations
        fprintf('Optimizing energy %d / %d\n',i,optimization_iterations);
        [x,fval] = opt(samples);
        if(fval < min_fval)
            min_fval = fval;
            best_x = x;
        end
    end

	
    fn = min(cur.NumberOfFrames, ref.NumberOfFrames);

    per_frame_ss = zeros(fn, 1);
    fprintf('==Computing final metric value==\n');
    for i = 1 : fn
        d.cur = applyt(xrgb2gray(im2double(read(cur, i))), best_x,center_prior);
        d.ref = xrgb2gray(im2double(read(ref,i)));

        per_frame_ss(i) = sum(min( d.cur(:) / sum(d.cur(:)), d.ref(:) / sum(d.ref(:)) ));
        if( mod(i,20) == 0)
            fprintf('%d / %d\n',i,fn);
        end
    end

    ss = sum( per_frame_ss(isfinite(per_frame_ss))) / size(per_frame_ss(:), 1);
end

function [res] = gt_dif(x, d)
    r = applyt(d.cur, x,d.center_prior);
    %res = mean( (d.ref(:)-r(:)).^2 );
	
	nr = reshape(r(:), size(r,2)*size(r,1), size(r,3));
	nr = nr ./ repmat(sum(nr, 1), [size(nr, 1), 1]);
	nr(isnan(nr)) = 0;
    
	nd = reshape(d.ref(:), size(d.ref,2)*size(d.ref,1), size(d.ref,3));
	nd = nd ./ repmat(sum(nd, 1), [size(nd, 1), 1]);
	nd(isnan(nd)) = 0;
    
	res = - mean(sum(min(nd, nr), 1));	
end

function [r] = applyt(A, x,center_prior)
    r = levels(A, x(2:6));
    
    r = r * (1.0 - x(1)) + repmat(center_prior, [1 1 size(r,3)]) * x(1);    
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
    fn = +Inf;
    if(~isempty(ref))
        fn = min(fn, ref.NumberOfFrames);
    end
    if(~isempty(cur))
        fn = min(fn, cur.NumberOfFrames);
    end    

    for i = 1:step:fn
       if(~isempty(ref))
        data.ref(:, :, floor((i - 1) / step) + 1) = xrgb2gray(im2double(read(ref, i)));
       end
       if(~isempty(cur))
        data.cur(:, :, floor((i - 1) / step) + 1) = xrgb2gray(im2double(read(cur, i)));  
       end
       fprintf('sampling %d / %d\n', i, fn);
    end
    
end

function [ x,fval ] = opt(data,nocp)

    lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    ub = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0];
    if(exist('nocp','var'))
        if(nocp)
            ub(1)=0;
        end
    end
    
    x0 = rand(1,6) .* (ub-lb) + lb;

    opt = optimset('Display', 'iter', 'UseParallel', 'always');
    [x,fval] = fmincon(@gt_dif, x0, [], [], [], [], lb, ub, [], opt, data);
end

function res = xrgb2gray(img)
    if ((sum(size(size(img))) == 4) && size(img,3) == 3)
        res = rgb2gray(img);
    else
        res = img;
    end
end


function cp = get_center_prior(gt_path,gt,sampling_density,optimization_iterations)
    cp_path = [gt_path , '_cp.png'];
    
    generate = true;
    if( exist(cp_path,'file') )
       cp_info = dir(cp_path);
       gt_info = dir(gt_path);
       
       if(cp_info.datenum > gt_info.datenum)
           generate = false;
       end
    end
    
    
    if(~generate)
        fprintf('Using precomputed center prior model\n');
        cp = imread(cp_path);
        cp = im2double(cp(:,:,1));
    else
        fprintf('Searching for the best center prior model\n');
        xx  = -gt.Width/2:gt.Width/2 - 1; sigmaX = gt.Width/7;
        yy  = -gt.Height/2:gt.Height/2 - 1; sigmaY = gt.Height/7;
       
        GX =  exp(-xx.^2 / (2*sigmaX^2));
        GY =  exp(-yy.^2 / (2*sigmaY^2));
        cp = GY' * GX;
        
        fprintf('Sampling ground-truth sequence\n');
        samples = sample([],gt,sampling_density);
        
        samples.cur = repmat(cp,[1 1 size(samples.ref,3) ] );
        samples.center_prior = cp;
        
        min_fval = +Inf;
    
        for i = 1:optimization_iterations
            fprintf('Optimizing energy %d / %d\n',i,optimization_iterations);
            [x,fval] = opt(samples,true);
            if(fval < min_fval)
                min_fval = fval;
                best_x = x;
            end
        end
        cp = applyt(cp,best_x,cp);
        imwrite(cp,cp_path);
    end
end