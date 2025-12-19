interface PageHeaderProps {
  title: string;
  subtitle: string;
  children?: React.ReactNode;
}

export default function PageHeader({
  title,
  subtitle,
  children,
}: PageHeaderProps) {
  return (
    <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-8 text-white">
      <h1 className="text-3xl font-bold mb-2">{title}</h1>
      <p className="text-purple-100 mb-6">{subtitle}</p>
      {children}
    </div>
  );
}
