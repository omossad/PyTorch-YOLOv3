function draw_heatmap(x,y)
%Read x,y coordinates of the fixation and plots the heatmap
%   This function reads the the x,y coordinates of the eye 
%   and plots different heatmap versions
%x = x';
%y = y';

% Bin the data:
pts = linspace(0, 8, 101);
N = histcounts2(y(:), x(:), pts, pts);

figure;
% Plot scattered data (for comparison):
%subplot(1, 2, 1);
%scatter(x, y, 'r.');
%axis equal;
%set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]));

% Plot heatmap:
%subplot(1, 2, 2);
imagesc(pts, pts, N);
axis equal;
set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]), 'YDir', 'normal');

% Normally distributed sample points:

% Bin the data:
pts = linspace(0, 8, 101);
N = histcounts2(y(:), x(:), pts, pts);

% Create Gaussian filter matrix:
[xG, yG] = meshgrid(-5:5);
sigma = 2.5;
g = exp(-xG.^2./(2.*sigma.^2)-yG.^2./(2.*sigma.^2));
g = g./sum(g(:));

figure;
% Plot scattered data (for comparison):
%subplot(1, 2, 1);
%scatter(x, y, 'r.');
%axis equal;
%set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]));

% Plot heatmap:
%subplot(1, 2, 2);
imagesc(pts, pts, conv2(N, g, 'same'));
axis equal;
set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]), 'YDir', 'normal');


% Generate grid and compute minimum distance:
pts = linspace(0, 8, 101);
[X, Y] = meshgrid(pts);
D = pdist2([x(:) y(:)], [X(:) Y(:)], 'euclidean', 'Smallest', 1);

%figure;
% Plot scattered data:
%subplot(1, 2, 1);
%scatter(x, y, 'r.');
%axis equal;
%set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]));

% Plot heatmap:
%subplot(1, 2, 2);
%imagesc(pts, pts, reshape(D, size(X)));
%axis equal;
%set(gca, 'XLim', pts([1 end]), 'YLim', pts([1 end]), 'YDir', 'normal');
%colormap(flip(parula(), 1));

end
 
