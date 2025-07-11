function [gbest, gbestval, convergence, fitcount] = TTSA_func(fhd, Dimension, Particle_Number, Max_Gen, VRmin, VRmax, varargin)
    tic; % 开始计时，用于测量算法运行时间
    % 初始化参数
    N = Particle_Number; % 粒子（树）的数量
    low = ceil(0.1 * N); % 每棵树产生的种子的最小数量
    high = ceil(0.25 * N); % 每棵树产生的种子的最大数量
    D = Dimension; % 问题的维度
    ST = 0.1; % 搜索倾向参数，用于决定种子生成策略
    dmin = VRmin; % 搜索空间的下界
    dmax = VRmax; % 搜索空间的上界
    fitcount = N; % 初始化函数评估次数为粒子数量
    trees = zeros(N, D); % 初始化树（粒子）的位置矩阵
    obj = zeros(1, N); % 初始化每棵树的目标函数值数组
    t = 1; % 初始化迭代计数器
    convergence = ones(1, Max_Gen); % 初始化收敛曲线数组

    % 使用伯努利映射初始化种群位置
    a = 0.4; % 伯努利映射的参数
    for i = 1:N
        bernoulli_values = bernoulli_map(zeros(1, D), D, a); % 使用伯努利映射初始化
        for j = 1:D
            trees(i, j) = dmin + (dmax - dmin) * bernoulli_values(j); % 将初始化的位置映射到搜索空间内
        end
    end

    % 评估初始种群的适应度
    obj = feval(fhd, trees', varargin{:}); % 使用目标函数评估每棵树的适应度

    % 寻找当前最佳树的位置
    [gbestval, best_idx] = min(obj); % 找到最小目标函数值及其索引
    gbest = trees(best_idx, :); % 获取最佳树的位置

    % 主循环
    while t <= Max_Gen
        [sorted_obj, sortIndex] = sort(obj); % 对目标函数值进行排序
        producer_ratio = 0.4 - 0.01 * t; % 计算生产者比例
        producer_num = max(1, ceil(producer_ratio * N)); % 计算生产者数量

        producers = trees(sortIndex(1:producer_num), :); % 获取生产者
        consumers = trees(sortIndex(producer_num:end), :); % 获取消费者
        ST = 1 - t / Max_Gen; % 更新搜索倾向参数

        for i = 1:N
            ns = low + randi(high - low + 1); % 为当前树生成随机数量的种子
            seeds = zeros(ns, D); % 初始化种子矩阵
            objs = zeros(1, ns); % 初始化种子的目标函数值数组
            for j = 1:ns
                r = fix(rand * N) + 1; % 随机选择另一棵树
                while i == r
                    r = fix(rand * N) + 1; % 确保不选择自己
                end
                beta = (0.5/Max_Gen)*t +1.5; % Equation 6   
                % 根据TLCO的位置更新公式修改种子位置生成方式
                if rand < ST
                    % TLCO中工蚁的位置更新公式
                    levy = levy_fun_TLCO(1,D,beta); % 生成Levy飞行步长
                    seeds(j, :) =  trees(i, :) + ...
                                   (-1+2*rand(1)).*(rand(1,D) + levy).*abs(gbest - trees(i, :)); % 更新种子位置
                else
                    % TLCO中兵蚁的位置更新公式
                    levy = levy_fun_TLCO(1,D,beta); % 生成Levy飞行步长
                    seeds(j, :) =2*rand(1).*gbest +...
                                  (-1+2*rand(1)).*abs( trees(i, :) - (levy.*gbest)) ;  % 更新种子位置  
                end
                seeds(j, :) = Bounds(seeds(j, :), dmin, dmax); % 确保种子位置在搜索空间内
            end
            objs = feval(fhd, seeds', varargin{:}); % 评估种子的适应度
            [val, seed_indis] = min(objs); % 找到最佳种子
            if objs(seed_indis) < obj(i)
                trees(i, :) = seeds(seed_indis, :); % 更新当前树的位置
                obj(i) = objs(seed_indis); % 更新当前树的目标函数值
            end

            % 蚁群差分进化策略
            CR = 0.9; % 交叉概率
            ant_idx1 = randi(N);
            ant_idx2 = randi(N);
            while ant_idx1 == ant_idx2
                ant_idx2 = randi(N);
            end
            ant_idx3 = randi(N);
            while ant_idx3 == ant_idx1 || ant_idx3 == ant_idx2
                ant_idx3 = randi(N);
            end

            % 生成差分向量
            AntMutant = trees(ant_idx1, :) + 0.5 * (trees(ant_idx2, :) - trees(ant_idx3, :));

            % 交叉操作
            ant_trial = trees(i, :);
            for d = 1:D
                if rand < CR
                    ant_trial(d) = AntMutant(d);
                end
            end

            % 确保ant_trial在边界内
            ant_trial = Bounds(ant_trial, dmin, dmax);

            % 计算ant_trial的适应度
            new_obj = feval(fhd, ant_trial', varargin{:});
            fitcount = fitcount + 1;

            % 选择操作
            if new_obj < obj(i)
                trees(i, :) = ant_trial;
                obj(i) = new_obj;
            end
        end

        % 检查种群多样性
        diversity = mean(std(trees, 0, 1));
        if diversity < 1e-3
            % 重新注入部分种群
            for i = 1:floor(0.2 * N)
                trees(sortIndex(end-i+1), :) = dmin + (dmax - dmin) * rand;
                obj(sortIndex(end-i+1)) = feval(fhd, trees(sortIndex(end-i+1), :)', varargin{:});
            end
        end

        % 引入正余弦策略
        for i = 1:N
            % 计算正余弦值
            A = 2 * (1 - t / Max_Gen); % 非线性递减因子
            r1 = rand();
            r2 = rand();
            a = A * cos(2 * pi * r1);
            b = A * sin(2 * pi * r2);
            new_position = trees(i, :) + a * (gbest - trees(i, :)) + b * (trees(randi(N), :) - trees(i, :));
            new_position = Bounds(new_position, dmin, dmax); % 确保新位置在边界内
            new_obj = feval(fhd, new_position', varargin{:});
            fitcount = fitcount + 1;
            if new_obj < obj(i)
                trees(i, :) = new_position;
                obj(i) = new_obj;
            end
        end

        % 更新全局最优解
        [current_best, current_best_idx] = min(obj);
        if current_best < gbestval
            gbestval = current_best; % 更新全局最佳目标函数值
            gbest = trees(current_best_idx, :); % 更新全局最佳位置
        end
        convergence(t) = gbestval; % 记录当前迭代的全局最佳目标函数值
        t = t + 1; % 更新迭代计数器
    end

    % 结束计时
    time = toc; % 测量算法运行时间
end

function s = Bounds(s, Lb, Ub)
    s(s < Lb) = Lb; % 如果位置小于下界，则设置为下界
    s(s > Ub) = Ub; % 如果位置大于上界，则设置为上界
end

function [z] = levy_fun_TLCO(n,m,beta)
    num = gamma(1+beta)*sin(pi*beta/2); % 用于分子的计算
    den = gamma((1+beta)/2)*beta*2^((beta-1)/2); % 用于分母的计算
    sigma_u = (num/den)^(1/beta);% 标准差
    u = random('Normal',0,sigma_u^2,n,m); 
    v = random('Normal',0,1,n,m);
    z =( u./(abs(v).^(1/beta)));
end

function bernoulli = bernoulli_map(trees, dim, a)
    % 伯努利映射函数
    bernoulli = rand(size(trees)); % 随机初始化
    for i = 1:size(trees, 1)
        for j = 2:dim
            if bernoulli(i,j-1) <= (1 - a)
                bernoulli(i,j) = bernoulli(i,j-1) / (1 - a);
            elseif bernoulli(i,j-1) > (1 - a)
                bernoulli(i,j) = (bernoulli(i,j-1) - 1 + a) / a;
            end
        end
    end
end