program test
    use polymer
    implicit none
    integer::n,i,j,m
    real *8,allocatable::results(:,:),arr(:,:),r(:)

    n = 10000
    m = 100
    allocate(arr(n,2),results(m,n))

    do i=1,m
        arr = genpoly(n)
        results(i,1) = sqrt(arr(1,1)**2+arr(1,2)**2)
        do j=2,n
            arr(j,:) = arr(j,:)+arr(j-1,:)
            results(i,j) = sqrt(sum(arr(j,:)**2))
        enddo
        
    enddo


    open(1,file='results.dat')
    do i=1,m
        do j=1,n
            write(1,*) results(j,i)
        enddo
    enddo
    close(1)
endprogram test
