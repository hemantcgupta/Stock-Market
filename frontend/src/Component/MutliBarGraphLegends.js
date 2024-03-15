import React, { Component } from 'react';


class MutliBarGraphLegends extends Component {
    constructor(props) {
        super(props);
        this.state = {
        }
    }

    componentDidMount() {
        // this.getCardsData()
    }

    render() {
        const { data, colors, agentsName, i } = this.props;
        return (
            i ? <div className='legends-box' style={{ width: `${agentsName && agentsName.length > 0 ? '320px' : '200px'}` }}>
                <div className="triangle-up"></div>
                <div id="legend" style={{ justifyContent: 'start', flexWrap: 'wrap' }}>
                    {data.length > 0 ? data[0].values.map((d, i) =>
                        <p className="sub-legend"><div className="square" style={{ background: colors[i] }}></div>{`${d.rate}`}</p>) : <p />}
                </div>
                <div className='category'>
                    {agentsName && agentsName.length > 0 ?
                        agentsName.map((d, i) =>
                            <p>{`${i + 1}. ${d.AgentName}`}</p>) :
                        data.map((d, i) =>
                            <p>{`${i + 1}. ${d.category}`}</p>)}
                </div>
            </div> :

                <div className="bottom-legends">
                    {data.length > 0 ? data[0].values.map((d, i) =>
                        <p className="sub-legend"><div className="square" style={{ background: colors[i] }}></div>{`${d.rate}`}</p>) : <p />}
                </div>
        );
    }
}
export default MutliBarGraphLegends;




