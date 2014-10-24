%This script perform small demo comparison of saliency models

%We start with downloading saliency sequences
%NOTE: this is just short fragment of data used in the research
%      full length ground truth saliency can be downloaded from
%      the project website http://compression.ru/video/savam/

if(~exist('example_data','file'))
    mkdir('example_data');
end
base_url = 'http://compression.ru/download/saliency/example/';

if(~exist('example/GT.mp4','file'))
    urlwrite([base_url , 'GT.mp4'],'example_data/GT.mp4');
end

if(~exist('example/One_human.mp4','file'))
    urlwrite([base_url , 'One_human.mp4'],'example_data/One_human.mp4');
end
 
if(~exist('example/Judd.mp4','file'))
    urlwrite([base_url , 'Judd.mp4'],'example_data/Judd.mp4');
end

if(~exist('example/src.mp4','file'))
    urlwrite([base_url , 'src.mp4'],'example_data/src.mp4');
end



%Now we measure score for each competitor

s = zeros(2,1);

s(1) = ss_robust_metric('example_data/Judd.mp4','example_data/GT.mp4',50,20);
s(2) = ss_robust_metric('example_data/One_human.mp4','example_data/GT.mp4',50,20);


%And show them  on plot with first frames
figure;

GT = VideoReader('example_data/GT.mp4');
Judd = VideoReader('example_data/Judd.mp4');
One_human = VideoReader('example_data/One_human.mp4');
src = VideoReader('example_data/src.mp4');

subplot(2,4,1);
imshow(src.read(10));
title('Source');


subplot(2,4,2);
imshow(GT.read(10));
title('GT');


subplot(2,4,3);
imshow(Judd.read(10));
title('Judd');


subplot(2,4,4);
imshow(One_human.read(10));
title('One human');


subplot(2,1,2)
bar(s);
set(gca,'XTickLabel',{'Judd','One human'});


